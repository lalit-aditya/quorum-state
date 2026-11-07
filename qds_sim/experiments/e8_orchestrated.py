"""Experiment 8 - The step-by-step framework with a small and an ideal model.

Implements the pipeline of the paper's framework figure and compares it, on the SAME
evidence, with traditional memories. Per session:

  Step 1  EXTRACT      small model reads the new evidence -> role, direction, strength
  Step 2  ROTATE       each item rotates the decision state (code, no tokens)
  Step 3  REVIEW       optional unsharp review of intensity s (code)
  Step 4  DRIFT        state relaxes toward the default |d> (code)
  Step 5  COORDINATE   per-role disagreement; optional proportional feedback (code)
  Step 6  ANSWER       ideal model reads ONLY the compact state description

Schemes (all see identical evidence; a decision is made at every --decide-every session
and always at the last one):

  transcript-ideal   ideal model re-reads the whole history at each decision  (traditional)
  transcript-small   small model re-reads the whole history at each decision  (cheaper traditional)
  retrieval-small    small model reads the 2 latest items per role            (cheapest baseline)
  state-code         small-model extraction + state; decision read from the state, no answer call
  orchestrated       small-model extraction + state; ideal model answers from the state (steps 1-6)

Measured: tokens reported by the API per call, dollars from --price-* (YOU must set these to
your real prices; the defaults are placeholders), accuracy against a synthetic ground truth
(approve iff the equal-weight sum of signed evidence strengths is positive), Brier score and
agreement with transcript-ideal.

Rules this script follows (see CLAUDE.md):
  * ground truth, prompts and scoring are NOT tuned to favour any scheme;
  * --state-scale defaults to 1.0. Changing it after seeing results on the same projects is
    post-hoc tuning: fix it in advance and use fresh --seed values;
  * a run is only valid with 0 parse errors, 0 truncated and 0 empty replies.

Backends:
  --backend mock       offline, token counts are ESTIMATES, the 'LLM' is a heuristic
  --backend anthropic  real calls; needs `pip install anthropic` and ANTHROPIC_API_KEY.
                       Prints the planned number of calls first; --yes skips the question.

Quick tour (no API):   python e8_orchestrated.py --demo
Real run (small):      python e8_orchestrated.py --backend anthropic --projects 3 --sessions 5 --yes
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from _common import binom_ci, out_dir, parse_args, plt, save_csv, save_json
from e6_agent_memory import ROLES, make_evidence
from e7_token_benchmark import (DECIDE, EXTRACT_BATCH, AnthropicBackend, MockBackend,
                                _extracted, parse_json_list, parse_p)
from qds import core

SCHEMES = ["transcript-ideal", "transcript-small", "retrieval-small", "state-code", "orchestrated"]
BASELINE = "transcript-ideal"
Y_AXIS = [0.0, 1.0, 0.0]


# --------------------------------------------------------------------------- #
# Accounting
# --------------------------------------------------------------------------- #
class SchemeUsage:
    def __init__(self):
        self.calls = 0
        self.inp = 0
        self.out = 0
        self.cost = 0.0

    def add(self, i, o, price):
        self.calls += 1
        self.inp += i
        self.out += o
        self.cost += (i * price[0] + o * price[1]) / 1e6

    def tokens(self):
        return self.inp + self.out


class Router:
    """Sends each prompt to the small or the ideal backend and charges the schemes
    that need it. Enforces one global call budget across both backends."""

    def __init__(self, small, ideal, prices, max_calls):
        self.b = {"small": small, "ideal": ideal}
        self.prices, self.max_calls, self.n = prices, max_calls, 0

    def ask(self, which, prompt, max_tokens, usage, charge):
        if self.n >= self.max_calls:
            raise RuntimeError(f"--max-calls {self.max_calls} reached; stopping to protect your budget")
        self.n += 1
        text, i, o = self.b[which].ask(prompt, max_tokens)
        for name in charge:
            usage[name].add(i, o, self.prices[which])
        return text


# --------------------------------------------------------------------------- #
# Decision state: steps 2-5 (plain code, no tokens)
# --------------------------------------------------------------------------- #
class DecisionState:
    def __init__(self, half_life=1e9, scale=1.0):
        prior = core.dm([np.cos(np.pi / 4), np.sin(np.pi / 4)])      # undecided prior
        self.rho = prior.copy()
        self.role_rho = {r: prior.copy() for r in ROLES}              # for coordination only
        self.P_approve = core.projector([1, 0])
        self.P_risk = core.projector([np.cos(np.pi / 8), np.sin(np.pi / 8)])
        self.half_life, self.scale = half_life, scale

    # step 2
    def rotate(self, role, sign, strength):
        U = core.rotation(ROLES[role], -sign * strength * self.scale)
        self.rho = core.evolve(self.rho, U)
        self.role_rho[role] = core.evolve(self.role_rho[role], U)

    # step 3
    def review(self, s):
        if s > 0:
            self.rho = core.apply_review(self.rho, self.P_approve, s)

    # step 4 (the default decision is 'do not approve', basis state |1>)
    def drift(self, dt):
        gamma = 1 - 0.5 ** (dt / self.half_life)
        self.rho = core.amplitude_damping(self.rho, gamma, default=1)
        for r in self.role_rho:
            self.role_rho[r] = core.amplitude_damping(self.role_rho[r], gamma, default=1)

    # step 5
    def coordinate(self, gain):
        p_roles = {r: core.prob(rho, self.P_approve) for r, rho in self.role_rho.items()}
        mean = float(np.mean(list(p_roles.values())))
        disagreement = float(np.std(list(p_roles.values())))
        if gain > 0:          # experimental proportional feedback toward the role consensus
            angle = -gain * (mean - self.p_approve())
            self.rho = core.evolve(self.rho, core.rotation(Y_AXIS, angle))
        return disagreement, p_roles

    def p_approve(self):
        return core.prob(self.rho, self.P_approve)

    def p_risk_ok(self):
        return core.prob(self.rho, self.P_risk)

    def describe(self, n_sessions):
        p = self.p_approve()
        return (f"Decision state after {n_sessions} sessions: P(approve)={p:.2f}, "
                f"P(risk acceptable)={self.p_risk_ok():.2f}; "
                f"{'leaning approve' if p >= 0.5 else 'leaning reject'}.")


# --------------------------------------------------------------------------- #
# One project
# --------------------------------------------------------------------------- #
def run_project(router, items, S, cfg, usage, log, verbose=False):
    truth = float(np.sum([it["sign"] * it["strength"] for it in items]) > 0)
    state = DecisionState(cfg.half_life, cfg.state_scale)
    final, trace = {}, []

    def say(step, msg):
        if verbose:
            print(f"  step {step}  {msg}")

    for s in range(S):
        sess = [it for it in items if it["session"] == s]
        seen = [it for it in items if it["session"] <= s]
        sess_txt = "\n".join("- " + it["text"] for it in sess)
        last = s == S - 1
        decide = ((s + 1) % cfg.decide_every == 0) or last
        if verbose:
            print(f"\nsession {s + 1}/{S}: {len(sess)} new evidence items")

        # step 1: small model extracts structure from the NEW items only
        text = router.ask("small", EXTRACT_BATCH + sess_txt, 60 * len(sess) + 40, usage,
                          ["state-code", "orchestrated"])
        try:
            objs = parse_json_list(text)
            if len(objs) != len(sess):
                raise ValueError(f"expected {len(sess)} objects, got {len(objs)}")
        except Exception as e:
            log.append(f"batch parse error: {e}")
            objs = [{} for _ in sess]
        parsed = [_extracted(d, it, log) for d, it in zip(objs, sess)]
        say(1, "extract : " + "; ".join(f"{r[:3]} {'+' if sg > 0 else '-'}{st:.2f}" for r, sg, st in parsed))

        # step 4 (elapsed time since the last session), then step 2 and step 3
        if s > 0:
            state.drift(1.0)
            say(4, f"drift   : half-life {cfg.half_life:g} sessions")
        for role, sign, strength in parsed:
            state.rotate(role, sign, strength)
        say(2, f"rotate  : P(approve) = {state.p_approve():.3f}")
        if cfg.review_every and (s + 1) % cfg.review_every == 0:
            state.review(cfg.review_intensity)
            say(3, f"review  : intensity {cfg.review_intensity} -> P(approve) = {state.p_approve():.3f}")

        # step 5
        disagreement, p_roles = state.coordinate(cfg.feedback_gain)
        say(5, f"coordinate: role disagreement {disagreement:.3f} "
               f"({', '.join(f'{r[:3]} {p:.2f}' for r, p in p_roles.items())})"
               f"{' with feedback gain ' + str(cfg.feedback_gain) if cfg.feedback_gain else ' (feedback off)'}")
        trace.append(dict(session=s, p_state=state.p_approve(), disagreement=disagreement))

        if not decide:
            continue

        # decisions of the traditional schemes
        t = router.ask("ideal", DECIDE + "All evidence so far:\n" + "\n".join("- " + it["text"] for it in seen),
                       120, usage, ["transcript-ideal"])
        p_ti = parse_p(t, log)
        t = router.ask("small", DECIDE + "All evidence so far:\n" + "\n".join("- " + it["text"] for it in seen),
                       120, usage, ["transcript-small"])
        p_ts = parse_p(t, log)
        picked = []
        for role in ROLES:
            picked += [it for it in seen if it["role"] == role][-2:]
        t = router.ask("small", DECIDE + "Most recent evidence per role:\n" + "\n".join("- " + it["text"] for it in picked),
                       120, usage, ["retrieval-small"])
        p_rs = parse_p(t, log)

        # step 6: the ideal model sees only the compact state
        desc = state.describe(s + 1)
        t = router.ask("ideal", DECIDE + desc, 120, usage, ["orchestrated"])
        p_or = parse_p(t, log)
        say(6, f"answer  : ideal model reads \"{desc}\" -> P(approve) = {p_or:.2f}")
        if last:
            final = {"transcript-ideal": p_ti, "transcript-small": p_ts, "retrieval-small": p_rs,
                     "state-code": state.p_approve(), "orchestrated": p_or}
    return truth, final, trace


def plan_calls(P, S, decide_every):
    D = sum(1 for s in range(S) if ((s + 1) % decide_every == 0) or s == S - 1)
    return P * (S + 4 * D), D


def main():
    def extra(p):
        p.add_argument("--backend", choices=["mock", "anthropic"], default="mock")
        p.add_argument("--small-model", default=os.environ.get("QDS_SMALL_MODEL", "claude-haiku-5-5"))
        p.add_argument("--ideal-model", default=os.environ.get("QDS_IDEAL_MODEL", "claude-sonnet-5-5"))
        p.add_argument("--price-small", default="1,5",
                       help="USD per million tokens 'input,output' for the small model. PLACEHOLDER: set your real price")
        p.add_argument("--price-ideal", default="3,15",
                       help="USD per million tokens 'input,output' for the ideal model. PLACEHOLDER: set your real price")
        p.add_argument("--projects", type=int, default=None)
        p.add_argument("--sessions", type=int, default=10)
        p.add_argument("--per-session", type=int, default=4)
        p.add_argument("--decide-every", type=int, default=1,
                       help="answer a query every N sessions (and at the last one); extraction still runs every session")
        p.add_argument("--half-life", type=float, default=1e9, help="drift half-life in sessions (default: no drift. The state still does NOT reproduce the additive ground truth: see e10)")
        p.add_argument("--state-scale", type=float, default=1.0, help="evidence strength multiplier. Keep 1.0 unless fixed in advance")
        p.add_argument("--review-every", type=int, default=0, help="unsharp review every N sessions (0 = off)")
        p.add_argument("--review-intensity", type=float, default=0.5)
        p.add_argument("--feedback-gain", type=float, default=0.0, help="experimental cross-role feedback (0 = off)")
        p.add_argument("--max-calls", type=int, default=2000)
        p.add_argument("--yes", action="store_true")
        p.add_argument("--demo", action="store_true", help="one short project, every step printed")
    args = parse_args(__doc__.splitlines()[0], extra)
    rng = np.random.default_rng(args.seed)
    if args.demo:
        args.projects, args.sessions = 1, min(args.sessions, 4)
    P = args.projects or (3 if args.quick else 10)
    S, E = args.sessions, args.per_session
    prices = {"small": tuple(float(x) for x in args.price_small.split(",")),
              "ideal": tuple(float(x) for x in args.price_ideal.split(","))}
    n_calls, D = plan_calls(P, S, args.decide_every)
    d = out_dir(args, "e8_orchestrated")
    if args.state_scale != 1.0:
        print(f"WARNING: --state-scale {args.state_scale}. If chosen after seeing results on these projects, "
              "this is post-hoc tuning; report it as exploratory.")

    if args.backend == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Set ANTHROPIC_API_KEY first.")
        print(f"Planned: {n_calls} API calls ({P} projects x {S} sessions; {D} decisions per project) "
              f"to {args.small_model} (small) and {args.ideal_model} (ideal).")
        if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
            sys.exit("Cancelled.")
        small = AnthropicBackend(args.small_model, args.max_calls)
        ideal = AnthropicBackend(args.ideal_model, args.max_calls)
    else:
        small, ideal = MockBackend(rng), MockBackend(rng)
        print(f"Mock backend: token counts are ESTIMATES ({n_calls} simulated calls). Prices are placeholders.")
    router = Router(small, ideal, prices, args.max_calls)

    usage = {s: SchemeUsage() for s in SCHEMES}
    rows, log, traces = [], [], []
    t0 = time.time()
    for proj in range(P):
        items = make_evidence(rng, S, E)
        before = {s: (usage[s].tokens(), usage[s].cost) for s in SCHEMES}
        if args.demo:
            print(f"project {proj + 1}: {len(items)} evidence items over {S} sessions")
        truth, final, trace = run_project(router, items, S, args, usage, log, verbose=args.demo)
        traces += [dict(project=proj, **r) for r in trace]
        for s in SCHEMES:
            rows.append(dict(project=proj, scheme=s, truth=truth, p_approve=final[s],
                             correct=int((final[s] >= 0.5) == bool(truth)),
                             agrees_with_baseline=int((final[s] >= 0.5) == (final[BASELINE] >= 0.5)),
                             brier=(final[s] - truth) ** 2,
                             tokens_this_project=usage[s].tokens() - before[s][0],
                             cost_this_project=usage[s].cost - before[s][1]))
        if not args.demo:
            print(f"project {proj + 1}/{P} done ({time.time() - t0:.0f}s)")

    summary = {}
    for s in SCHEMES:
        rr = [r for r in rows if r["scheme"] == s]
        k = sum(r["correct"] for r in rr)
        lo, hi = binom_ci(k, len(rr))
        summary[s] = dict(calls=usage[s].calls, input_tokens=usage[s].inp, output_tokens=usage[s].out,
                          total_tokens=usage[s].tokens(), tokens_per_project=usage[s].tokens() / P,
                          cost_usd=usage[s].cost, cost_per_project=usage[s].cost / P,
                          accuracy=k / len(rr), accuracy_ci95=[lo, hi],
                          brier=float(np.mean([r["brier"] for r in rr])),
                          agreement_with_baseline=float(np.mean([r["agrees_with_baseline"] for r in rr])))
    bt, bc = summary[BASELINE]["total_tokens"], summary[BASELINE]["cost_usd"]
    for s in SCHEMES:
        summary[s]["tokens_vs_baseline"] = summary[s]["total_tokens"] / bt if bt else float("nan")
        summary[s]["cost_vs_baseline"] = summary[s]["cost_usd"] / bc if bc else float("nan")
        summary[s]["bill_if_baseline_is_100"] = 100 * summary[s]["cost_vs_baseline"]
    valid = (len(log) == 0 and small.truncated + ideal.truncated == 0 and small.empty + ideal.empty == 0)
    meta = dict(backend=args.backend,
                small_model=args.small_model if args.backend == "anthropic" else "mock",
                ideal_model=args.ideal_model if args.backend == "anthropic" else "mock",
                token_counts="estimated" if small.estimated else "reported by API",
                prices_usd_per_mtok=prices, prices_are_placeholders=(args.price_small == "1,5" and args.price_ideal == "3,15"),
                projects=P, sessions=S, items_per_session=E, decide_every=args.decide_every,
                half_life=args.half_life, state_scale=args.state_scale, review_every=args.review_every,
                feedback_gain=args.feedback_gain, parse_errors=len(log),
                truncated_replies=small.truncated + ideal.truncated, empty_replies=small.empty + ideal.empty,
                valid=valid, total_calls=router.n)
    save_csv(f"{d}/per_project.csv", rows)
    save_csv(f"{d}/state_trace.csv", traces)
    save_json(f"{d}/summary.json", dict(meta=meta, schemes=summary, parse_errors=log[:50]))

    fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.6))
    for ax, key, lab in zip(axs, ["tokens_per_project", "cost_per_project", "accuracy"],
                            ["tokens per project", "USD per project", "accuracy vs ground truth"]):
        ax.bar(range(len(SCHEMES)), [summary[s][key] for s in SCHEMES], color="gray" if key != "accuracy" else "C0")
        ax.set_xticks(range(len(SCHEMES)))
        ax.set_xticklabels(SCHEMES, rotation=30, ha="right", fontsize=6)
        ax.set_ylabel(lab)
    axs[2].set_ylim(0, 1)
    fig.suptitle(f"{meta['small_model']} + {meta['ideal_model']} - tokens {meta['token_counts']}", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{d}/orchestration.png")

    print(f"\n[small={meta['small_model']}, ideal={meta['ideal_model']}; tokens {meta['token_counts']}; "
          f"{P} projects x {S} sessions; decide every {args.decide_every}]")
    print(f"{'scheme':17s} {'calls':>6s} {'tokens':>9s} {'tok/base':>8s} {'USD':>8s} {'$/base':>7s} "
          f"{'accuracy [95% CI]':>22s} {'Brier':>6s} {'agree':>6s}")
    for s in SCHEMES:
        v = summary[s]
        ci = f"{v['accuracy']:.2f} [{v['accuracy_ci95'][0]:.2f},{v['accuracy_ci95'][1]:.2f}]"
        print(f"{s:17s} {v['calls']:6d} {v['total_tokens']:9d} {v['tokens_vs_baseline']:8.2f} "
              f"{v['cost_usd']:8.4f} {v['cost_vs_baseline']:7.2f} {ci:>22s} {v['brier']:6.3f} "
              f"{v['agreement_with_baseline']:6.2f}")
    print(f"\nIf your {BASELINE} bill were $100 (all of it memory cost): "
          + ", ".join(f"{s} ~ ${summary[s]['bill_if_baseline_is_100']:.0f}" for s in SCHEMES))
    if meta["prices_are_placeholders"]:
        print("NOTE: prices are the placeholder defaults. Set --price-small / --price-ideal to your real prices.")
    if not valid:
        print(f"\n*** RUN NOT VALID: {len(log)} parse errors, {meta['truncated_replies']} truncated and "
              f"{meta['empty_replies']} empty replies (see summary.json). Do not use these numbers. ***")
    else:
        print("\nRun valid: 0 parse errors, 0 truncated, 0 empty replies.")
    print(f"outputs -> {d}")


if __name__ == "__main__":
    main()
