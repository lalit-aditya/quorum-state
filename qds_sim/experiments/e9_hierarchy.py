"""Experiment 9 - Macro / meso / micro hierarchy with a small and an ideal model.

Extends experiment 8 (the step-by-step pipeline) to the three planning levels of the paper
(Section 3.1). A construction project is decided at three levels:

  macro  the goal      approve | reject
  meso   the plan      fast-track | phased | none (none if rejected)
  micro  the action    commit | defer      (commit the budget)

Per session: a small model extracts (role, direction, strength) from the NEW evidence only;
each level keeps its own qubit decision state and the relevant roles rotate it:

  macro state <- evidence of ALL roles          meso state <- engineering + operations
  micro state <- finance                        (plain code, no tokens)

The macro outcome k prepares the meso state (W_k rho_ref W_k^dagger, paper Eq. 1) and the meso
outcome prepares the micro state. In this experiment W_k are FIXED (identity unless
--prep-bias is set), not learned: there is no data to learn them from. Optional micro -> meso
feedback (--feedback-eps) is experimental and OFF by default.

!!  THE GROUND TRUTH BELOW IS SYNTHETIC. It is a rule written by the script author, not real
!!  construction logic, and it favours additive memories. Replace `ground_truth` with your
!!  real rule before drawing any practical conclusion.

    macro = sum of all signed evidence strengths > 0
    meso  = 'none' if not macro else 'fast-track' if (engineering + operations sum) > 0 else 'phased'
    micro = 'commit' if macro and (finance sum) > 0 else 'defer'

Schemes (identical evidence; a decision at every --decide-every session and at the last):

  flat-ideal    ideal model re-reads the whole history, answers all three levels in one call
  flat-small    same with the small model
  hier-code     small-model extraction + three level states; answers read from the states (no answer call)
  hier-ideal    same states; the ideal model answers all three levels from a one-line state description
  hier-gated    same states; code answers confident levels, and the ideal model is called ONLY when some
                level is uncertain (|p - 0.5| < --escalate-margin); it then overrides just the uncertain levels

NOTE: 'hier-gated' (uncertainty-gated escalation) is an extra design choice of this script, NOT
something the paper defines. The paper's rule "re-examine the macro decision when the meso
probability of reaching the goal falls below a threshold" is NOT implemented, because "meso
probability of reaching the goal" has not been operationalised yet.

Backends:  --backend mock (offline, estimated tokens)  |  --backend anthropic (needs ANTHROPIC_API_KEY)
Quick tour: python e9_hierarchy.py --demo
"""
from __future__ import annotations

import os
import re
import sys
import time

import numpy as np

from _common import binom_ci, out_dir, parse_args, plt, save_csv, save_json
from e6_agent_memory import ROLES, make_evidence
from e7_token_benchmark import (EXTRACT, EXTRACT_BATCH, AnthropicBackend, MockBackend,
                                _extracted, parse_json, parse_json_list)
from e8_orchestrated import DecisionState, Router, SchemeUsage
from qds import core

SCHEMES = ["flat-ideal", "flat-small", "hier-code", "hier-ideal", "hier-gated"]
BASELINE = "flat-ideal"
LEVELS = ["macro", "meso", "micro"]
LEVEL_ROLES = {"macro": ("finance", "engineering", "operations"),
               "meso": ("engineering", "operations"),
               "micro": ("finance",)}
VALID = {"macro": ("approve", "reject"), "meso": ("fast-track", "phased", "none"), "micro": ("commit", "defer")}
Y_AXIS = [0.0, 1.0, 0.0]

HDECIDE = ("You are deciding on a construction project from the evidence below.\n"
           "Rules: macro = 'approve' if the overall evidence supports the project, else 'reject'. "
           "meso = if you approve, 'fast-track' only if the engineering and operations evidence together "
           "support moving quickly, otherwise 'phased'; if you reject, 'none'. "
           "micro = 'commit' (commit the budget) only if you approve and the finance evidence supports "
           "spending, otherwise 'defer'.\n"
           'Reply with JSON only: {"macro": "approve"|"reject", "meso": "fast-track"|"phased"|"none", '
           '"micro": "commit"|"defer"}.\n\n')


# --------------------------------------------------------------------------- #
# Synthetic ground truth (REPLACE with your real rule)
# --------------------------------------------------------------------------- #
def ground_truth(items):
    tot = lambda roles: float(sum(it["sign"] * it["strength"] for it in items if it["role"] in roles))
    macro = tot(ROLES) > 0
    meso = "none" if not macro else ("fast-track" if tot(("engineering", "operations")) > 0 else "phased")
    micro = "commit" if (macro and tot(("finance",)) > 0) else "defer"
    return dict(macro="approve" if macro else "reject", meso=meso, micro=micro)


# --------------------------------------------------------------------------- #
# Three-level state (plain code, no tokens)
# --------------------------------------------------------------------------- #
class HierState:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lv = {k: DecisionState(cfg.half_life, cfg.state_scale) for k in LEVELS}
        self.prepare_meso(True)

    def prepare_meso(self, approve):
        """Macro outcome k prepares the meso state: W_k rho_ref W_k^dagger (W_k fixed, not learned)."""
        W = core.rotation(Y_AXIS, -self.cfg.prep_bias) if (approve and self.cfg.prep_bias) else np.eye(2)
        rho_ref = self.lv["meso"].rho
        self.lv["meso"].rho = W @ rho_ref @ W.conj().T

    def drift(self, dt):
        for st in self.lv.values():
            st.drift(dt)

    def rotate(self, role, sign, strength):
        for k in LEVELS:
            if role in LEVEL_ROLES[k]:
                self.lv[k].rotate(role, sign, strength)

    def feedback(self):
        """Experimental micro -> meso feedback: a small rotation proportional to the micro leaning."""
        eps = self.cfg.feedback_eps
        if eps > 0:
            lean = self.lv["micro"].p_approve() - 0.5
            self.lv["meso"].rho = core.evolve(self.lv["meso"].rho, core.rotation(Y_AXIS, -eps * lean))

    def p(self):
        return dict(macro=self.lv["macro"].p_approve(), meso=self.lv["meso"].p_approve(),
                    micro=self.lv["micro"].p_approve())

    def answer(self):
        p = self.p()
        macro = "approve" if p["macro"] >= 0.5 else "reject"
        meso = "none" if macro == "reject" else ("fast-track" if p["meso"] >= 0.5 else "phased")
        micro = "commit" if (macro == "approve" and p["micro"] >= 0.5) else "defer"
        return dict(macro=macro, meso=meso, micro=micro)

    def uncertain(self):
        p, m = self.p(), self.cfg.escalate_margin
        out = {k: abs(p[k] - 0.5) < m for k in LEVELS}
        if self.answer()["macro"] == "reject":          # meso is 'none' whatever its state says
            out["meso"] = False
        return out

    def describe(self, n):
        p = self.p()
        return (f"Decision state after {n} sessions (probability of the favourable answer at each level): "
                f"macro approve={p['macro']:.2f}; meso fast-track if approved={p['meso']:.2f}; "
                f"micro commit budget if approved={p['micro']:.2f}.")


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_levels(text, log):
    try:
        d = parse_json(text)
        out = {k: str(d[k]).strip().lower() for k in LEVELS}
        for k in LEVELS:
            if out[k] not in VALID[k]:
                raise ValueError(f"bad {k}: {out[k]!r}")
        return out
    except Exception as e:
        log.append(f"decision parse error: {e} :: {text[:100]!r}")
        return dict(macro="reject", meso="none", micro="defer")


class HierMock(MockBackend):
    """Offline stand-in that also understands the hierarchical decision prompt."""

    def ask(self, prompt, max_tokens=120):
        if not prompt.startswith(HDECIDE[:30]):
            return super().ask(prompt, max_tokens)
        body = prompt[len(HDECIDE):]
        m = re.search(r"macro approve=(\d\.\d+); meso fast-track if approved=(\d\.\d+); micro commit budget if approved=(\d\.\d+)", body)
        if m:
            pm, ps, pc = (float(x) for x in m.groups())
            macro = "approve" if pm >= 0.5 else "reject"
            meso = "none" if macro == "reject" else ("fast-track" if ps >= 0.5 else "phased")
            micro = "commit" if (macro == "approve" and pc >= 0.5) else "defer"
        else:
            tot = {r: 0.0 for r in ROLES}
            for line in body.splitlines():
                if line.startswith("- "):
                    d = parse_json(super().ask(EXTRACT + line[2:])[0])
                    tot[d["role"]] += d["direction"] * d["strength"]
            macro = "approve" if sum(tot.values()) > 0 else "reject"
            meso = "none" if macro == "reject" else ("fast-track" if tot["engineering"] + tot["operations"] > 0 else "phased")
            micro = "commit" if (macro == "approve" and tot["finance"] > 0) else "defer"
        out = f'{{"macro": "{macro}", "meso": "{meso}", "micro": "{micro}"}}'
        return out, self._tok(prompt), self._tok(out)


# --------------------------------------------------------------------------- #
# One project
# --------------------------------------------------------------------------- #
def run_project(router, items, S, cfg, usage, log, esc, verbose=False):
    truth = ground_truth(items)
    state = HierState(cfg)
    final = {}

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

        # step 1: small model extracts from the NEW items only
        text = router.ask("small", EXTRACT_BATCH + sess_txt, 60 * len(sess) + 40, usage,
                          ["hier-code", "hier-ideal", "hier-gated"])
        try:
            objs = parse_json_list(text)
            if len(objs) != len(sess):
                raise ValueError(f"expected {len(sess)} objects, got {len(objs)}")
        except Exception as e:
            log.append(f"batch parse error: {e}")
            objs = [{} for _ in sess]
        parsed = [_extracted(d, it, log) for d, it in zip(objs, sess)]
        say(1, "extract : " + "; ".join(f"{r[:3]} {'+' if sg > 0 else '-'}{st:.2f}" for r, sg, st in parsed))

        # drift (elapsed time), rotate each level's state, optional micro -> meso feedback
        if s > 0:
            state.drift(1.0)
        for role, sign, strength in parsed:
            state.rotate(role, sign, strength)
        state.feedback()
        p = state.p()
        say(2, f"levels  : macro {p['macro']:.2f} | meso {p['meso']:.2f} | micro {p['micro']:.2f}   "
               f"-> code answer {state.answer()}")
        if not decide:
            continue

        # traditional schemes: one call, all three levels
        hist = "All evidence so far:\n" + "\n".join("- " + it["text"] for it in seen)
        a_fi = parse_levels(router.ask("ideal", HDECIDE + hist, 160, usage, ["flat-ideal"]), log)
        a_fs = parse_levels(router.ask("small", HDECIDE + hist, 160, usage, ["flat-small"]), log)

        # state schemes
        code_ans = state.answer()
        desc = state.describe(s + 1)
        a_hi = parse_levels(router.ask("ideal", HDECIDE + desc, 160, usage, ["hier-ideal"]), log)
        unc = state.uncertain()
        esc["decisions"] += 1
        if any(unc.values()):
            a_ideal = parse_levels(router.ask("ideal", HDECIDE + desc, 160, usage, ["hier-gated"]), log)
            a_hg = {k: (a_ideal[k] if unc[k] else code_ans[k]) for k in LEVELS}
            esc["escalated"] += 1
            for k in LEVELS:
                esc[k] += int(unc[k])
            say(6, f"gate    : uncertain {[k for k in LEVELS if unc[k]]} -> ideal model called")
        else:
            a_hg = code_ans
            say(6, "gate    : all levels confident -> answered by code, no model call")
        if last:
            final = {"flat-ideal": a_fi, "flat-small": a_fs, "hier-code": code_ans,
                     "hier-ideal": a_hi, "hier-gated": a_hg}
            say(6, f"answer  : truth {truth}")
    return truth, final


def plan_calls(P, S, decide_every):
    D = sum(1 for s in range(S) if ((s + 1) % decide_every == 0) or s == S - 1)
    return P * (S + 3 * D) + P * D, D        # last term: upper bound for the gated escalations


def main():
    def extra(p):
        p.add_argument("--backend", choices=["mock", "anthropic"], default="mock")
        p.add_argument("--small-model", default=os.environ.get("QDS_SMALL_MODEL", "claude-haiku-5-5"))
        p.add_argument("--ideal-model", default=os.environ.get("QDS_IDEAL_MODEL", "claude-sonnet-5-5"))
        p.add_argument("--price-small", default="1,5", help="USD per Mtok 'in,out'. PLACEHOLDER: set your real price")
        p.add_argument("--price-ideal", default="3,15", help="USD per Mtok 'in,out'. PLACEHOLDER: set your real price")
        p.add_argument("--projects", type=int, default=None)
        p.add_argument("--sessions", type=int, default=10)
        p.add_argument("--per-session", type=int, default=4)
        p.add_argument("--decide-every", type=int, default=1)
        p.add_argument("--escalate-margin", type=float, default=0.10, help="hier-gated: call the ideal model when |p-0.5| is below this")
        p.add_argument("--half-life", type=float, default=1e9)
        p.add_argument("--state-scale", type=float, default=1.0, help="keep 1.0 unless fixed in advance")
        p.add_argument("--prep-bias", type=float, default=0.0, help="W_k rotation angle for the meso preparation (fixed, not learned)")
        p.add_argument("--feedback-eps", type=float, default=0.0, help="experimental micro->meso feedback (0 = off)")
        p.add_argument("--max-calls", type=int, default=2500)
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
    d = out_dir(args, "e9_hierarchy")
    if args.state_scale != 1.0:
        print(f"WARNING: --state-scale {args.state_scale}: post-hoc tuning if chosen after seeing these projects.")
    print("NOTE: the ground truth is a SYNTHETIC rule (see the docstring). Replace it with your real rule.")

    if args.backend == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Set ANTHROPIC_API_KEY first.")
        print(f"Planned: up to {n_calls} API calls ({P} projects x {S} sessions; {D} decisions per project) "
              f"to {args.small_model} (small) and {args.ideal_model} (ideal).")
        if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
            sys.exit("Cancelled.")
        small = AnthropicBackend(args.small_model, args.max_calls)
        ideal = AnthropicBackend(args.ideal_model, args.max_calls)
    else:
        small, ideal = HierMock(rng), HierMock(rng)
        print(f"Mock backend: token counts are ESTIMATES (up to {n_calls} simulated calls). Prices are placeholders.")
    router = Router(small, ideal, prices, args.max_calls)

    usage = {s: SchemeUsage() for s in SCHEMES}
    esc = dict(decisions=0, escalated=0, macro=0, meso=0, micro=0)
    rows, log = [], []
    t0 = time.time()
    for proj in range(P):
        items = make_evidence(rng, S, E)
        before = {s: (usage[s].tokens(), usage[s].cost) for s in SCHEMES}
        if args.demo:
            print(f"project {proj + 1}: {len(items)} evidence items over {S} sessions")
        truth, final = run_project(router, items, S, args, usage, log, esc, verbose=args.demo)
        for s in SCHEMES:
            r = dict(project=proj, scheme=s, tokens_this_project=usage[s].tokens() - before[s][0],
                     cost_this_project=usage[s].cost - before[s][1])
            for k in LEVELS:
                r[f"truth_{k}"], r[f"ans_{k}"] = truth[k], final[s][k]
                r[f"ok_{k}"] = int(final[s][k] == truth[k])
            r["ok_all"] = int(all(r[f"ok_{k}"] for k in LEVELS))
            rows.append(r)
        if not args.demo:
            print(f"project {proj + 1}/{P} done ({time.time() - t0:.0f}s)")

    summary = {}
    for s in SCHEMES:
        rr = [r for r in rows if r["scheme"] == s]
        v = dict(calls=usage[s].calls, input_tokens=usage[s].inp, output_tokens=usage[s].out,
                 total_tokens=usage[s].tokens(), tokens_per_project=usage[s].tokens() / P,
                 cost_usd=usage[s].cost, cost_per_project=usage[s].cost / P)
        for key in LEVELS + ["all"]:
            k = sum(r[f"ok_{key}"] for r in rr)
            lo, hi = binom_ci(k, len(rr))
            v[f"acc_{key}"], v[f"acc_{key}_ci95"] = k / len(rr), [lo, hi]
        summary[s] = v
    bt, bc = summary[BASELINE]["total_tokens"], summary[BASELINE]["cost_usd"]
    for s in SCHEMES:
        summary[s]["tokens_vs_baseline"] = summary[s]["total_tokens"] / bt if bt else float("nan")
        summary[s]["cost_vs_baseline"] = summary[s]["cost_usd"] / bc if bc else float("nan")
        summary[s]["bill_if_baseline_is_100"] = 100 * summary[s]["cost_vs_baseline"]
    esc_rate = esc["escalated"] / esc["decisions"] if esc["decisions"] else float("nan")
    valid = (len(log) == 0 and small.truncated + ideal.truncated == 0 and small.empty + ideal.empty == 0)
    meta = dict(backend=args.backend, small_model=args.small_model if args.backend == "anthropic" else "mock",
                ideal_model=args.ideal_model if args.backend == "anthropic" else "mock",
                token_counts="estimated" if small.estimated else "reported by API",
                prices_usd_per_mtok=prices,
                prices_are_placeholders=(args.price_small == "1,5" and args.price_ideal == "3,15"),
                ground_truth="synthetic rule written by the script author", projects=P, sessions=S,
                items_per_session=E, decide_every=args.decide_every, escalate_margin=args.escalate_margin,
                escalation=dict(esc, rate=esc_rate), state_scale=args.state_scale, prep_bias=args.prep_bias,
                feedback_eps=args.feedback_eps, parse_errors=len(log),
                truncated_replies=small.truncated + ideal.truncated, empty_replies=small.empty + ideal.empty,
                valid=valid, total_calls=router.n)
    save_csv(f"{d}/per_project.csv", rows)
    save_json(f"{d}/summary.json", dict(meta=meta, schemes=summary, parse_errors=log[:50]))

    fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.6))
    for ax, key, lab in zip(axs, ["tokens_per_project", "cost_per_project", "acc_all"],
                            ["tokens per project", "USD per project", "all three levels correct"]):
        ax.bar(range(len(SCHEMES)), [summary[s][key] for s in SCHEMES], color="C0" if key == "acc_all" else "gray")
        ax.set_xticks(range(len(SCHEMES)))
        ax.set_xticklabels(SCHEMES, rotation=30, ha="right", fontsize=6)
        ax.set_ylabel(lab)
    axs[2].set_ylim(0, 1)
    fig.suptitle(f"{meta['small_model']} + {meta['ideal_model']} - tokens {meta['token_counts']}", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{d}/hierarchy.png")

    print(f"\n[small={meta['small_model']}, ideal={meta['ideal_model']}; tokens {meta['token_counts']}; "
          f"{P} projects x {S} sessions; decide every {args.decide_every}]")
    print(f"{'scheme':11s} {'calls':>6s} {'tokens':>9s} {'tok/base':>8s} {'USD':>8s} {'$/base':>7s} "
          f"{'macro':>6s} {'meso':>6s} {'micro':>6s} {'all 3 [95% CI]':>18s}")
    for s in SCHEMES:
        v = summary[s]
        ci = f"{v['acc_all']:.2f} [{v['acc_all_ci95'][0]:.2f},{v['acc_all_ci95'][1]:.2f}]"
        print(f"{s:11s} {v['calls']:6d} {v['total_tokens']:9d} {v['tokens_vs_baseline']:8.2f} {v['cost_usd']:8.4f} "
              f"{v['cost_vs_baseline']:7.2f} {v['acc_macro']:6.2f} {v['acc_meso']:6.2f} {v['acc_micro']:6.2f} {ci:>18s}")
    print(f"\nhier-gated escalated {esc['escalated']} of {esc['decisions']} decisions "
          f"({100 * esc_rate:.0f}%): macro {esc['macro']}, meso {esc['meso']}, micro {esc['micro']}")
    print(f"If your {BASELINE} bill were $100 (all of it memory cost): "
          + ", ".join(f"{s} ~ ${summary[s]['bill_if_baseline_is_100']:.0f}" for s in SCHEMES))
    if meta["prices_are_placeholders"]:
        print("NOTE: prices are placeholders. Set --price-small / --price-ideal to your real prices.")
    if not valid:
        print(f"\n*** RUN NOT VALID: {len(log)} parse errors, {meta['truncated_replies']} truncated and "
              f"{meta['empty_replies']} empty replies (see summary.json). Do not use these numbers. ***")
    else:
        print("\nRun valid: 0 parse errors, 0 truncated, 0 empty replies.")
    print(f"outputs -> {d}")


if __name__ == "__main__":
    main()
