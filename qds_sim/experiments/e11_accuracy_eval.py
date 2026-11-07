"""Experiment 11 - Accuracy evaluation of the memory schemes, with fair controls.

ONE script that answers: "is the decision state as ACCURATE as the traditional memories,
and where does it (not) pay off?"  It does what the e8 runs could not:

  * enough projects for a real comparison (default 30; use 100+ for a paper claim),
  * truths balanced 50/50 by construction (rejection sampling), so "always approve" scores 0.5,
  * TWO ground truths:
        additive  approve iff the equal-weight sum of signed strengths > 0      (as in e7/e8)
        order     approve iff a RECENCY-WEIGHTED sum > 0: z <- 0.9*z + sign*strength, item by
                  item in time order. The answer depends on evidence ORDER. The rule is stated
                  in every prompt of the LLM schemes, in the same words, so no scheme is
                  surprised by it.   [REPLACE with your real decision rule if you have one]
  * accuracy at several history lengths (--checkpoints), from one run,
  * control schemes that cost no tokens, to separate WHY a scheme errs:
        rule-code         applies the true rule to the true items (must be 100%; sanity check)
        additive-code     equal-weight sum of the true items       (what an additive memory could do)
        state-exact-code  the quantum state fed the TRUE role/direction/strength (no extraction
                          noise): its error is purely the state's functional form
        state-code        the state fed the LLM-extracted values: adds extraction error
  * paired tests against the baseline (exact two-sided binomial on discordant projects),
  * Brier score, confidence when wrong, accuracy on clear vs close projects,
  * per-scheme tokens, OUTPUT tokens per call (so a reasoning-length difference between schemes
    cannot hide inside the cost) and cost, with the thinking setting recorded.

LLM schemes (all see identical evidence):
  transcript-ideal  ideal model re-reads everything             (baseline)
  transcript-small  small model re-reads everything
  retrieval-small   small model reads the 2 latest items per role
  summary-small     small model keeps a 60-word running summary, then answers   (--with-summary)
  state-code        small-model extraction -> state -> decision read in code
  orchestrated      small-model extraction -> state -> ideal model answers from a one-line state

Rules (CLAUDE.md): nothing is tuned after seeing results; --state-scale stays 1.0 unless fixed
in advance; use fresh --seed values for confirmation runs; a run is valid only with 0 parse
errors, 0 truncated and 0 empty replies. Prices are placeholders unless you set them.

Offline check (no API):
    python e11_accuracy_eval.py --projects 6 --sessions 6 --checkpoints 3,6 --with-summary
Real run (set ANTHROPIC_API_KEY first; the plan and the call count are printed before any call):
    python e11_accuracy_eval.py --backend anthropic --projects 60 --sessions 30 --checkpoints 10,30 \
        --truth both --ideal-thinking off --price-small 1,5 --price-ideal 3,15 --max-calls 8000
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from _common import binom_ci, out_dir, parse_args, plt, save_csv, save_json
from e6_agent_memory import ROLES, make_evidence
from e7_token_benchmark import (DECIDE, EXTRACT_BATCH, SUMMARISE, AnthropicBackend, MockBackend,
                                _extracted, parse_json_list, parse_p)
from e8_orchestrated import DecisionState, Router, SchemeUsage

DECAY = 0.9
RULE_TEXT = ("Rule: the evidence is listed oldest first and recent evidence counts more. Each item "
             "counts 10% less than the item after it. Approve if the weighted total of the signed "
             "strengths is positive.")
BASELINE = "transcript-ideal"
LLM_SCHEMES = ["transcript-ideal", "transcript-small", "retrieval-small", "state-code", "orchestrated"]
CODE_SCHEMES = ["rule-code", "additive-code", "state-exact-code"]


# --------------------------------------------------------------------------- #
# Ground truths (on the TRUE items; never shown to any scheme except as the stated rule)
# --------------------------------------------------------------------------- #
def truth_additive(items):
    z = float(sum(it["sign"] * it["strength"] for it in items))
    return float(z > 0), abs(z)


def truth_order(items):
    z = 0.0
    for it in items:
        z = DECAY * z + it["sign"] * it["strength"]
    return float(z > 0), abs(z)


TRUTHS = {"additive": truth_additive, "order": truth_order}


def balanced_items(rng, S, E, want, truth_fn, tries=400):
    """Draw evidence until the FINAL truth equals `want` (keeps the classes 50/50)."""
    items = None
    for _ in range(tries):
        items = make_evidence(rng, S, E)
        if truth_fn(items)[0] == want:
            return items
    return items


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
def rule_prefix(mode):
    return (RULE_TEXT + "\n\n") if mode == "order" else ""


def bullets(its):
    return "\n".join("- " + it["text"] for it in its)


# --------------------------------------------------------------------------- #
# One project: all schemes, all checkpoints
# --------------------------------------------------------------------------- #
def run_project(router, items, S, args, mode, usage, log, checkpoints, with_summary):
    truth_fn = TRUTHS[mode]
    state = DecisionState(args.half_life, args.state_scale)       # fed LLM-extracted values
    exact = DecisionState(args.half_life, args.state_scale)       # fed the TRUE values
    summary = ""
    out = {}                                                      # checkpoint -> dict(truth, margin, p per scheme)
    for s in range(S):
        sess = [it for it in items if it["session"] == s]
        seen = [it for it in items if it["session"] <= s]
        sess_txt = bullets(sess)

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
        if s > 0:
            state.drift(1.0)
            exact.drift(1.0)
        for role, sign, strength in parsed:
            state.rotate(role, sign, strength)
        for it in sess:
            exact.rotate(it["role"], it["sign"], it["strength"])

        if with_summary:
            t = router.ask("small", SUMMARISE + rule_prefix(mode) + "Current summary:\n" + (summary or "(none)")
                           + "\n\nNew evidence:\n" + sess_txt, 150, usage, ["summary-small"])
            summary = t.strip()

        if (s + 1) not in checkpoints:
            continue

        truth, margin = truth_fn(seen)
        pre = rule_prefix(mode)
        p = {}
        t = router.ask("ideal", DECIDE + pre + "All evidence so far (oldest first):\n" + bullets(seen),
                       120, usage, ["transcript-ideal"])
        p["transcript-ideal"] = parse_p(t, log)
        t = router.ask("small", DECIDE + pre + "All evidence so far (oldest first):\n" + bullets(seen),
                       120, usage, ["transcript-small"])
        p["transcript-small"] = parse_p(t, log)
        picked = []
        for role in ROLES:
            picked += [it for it in seen if it["role"] == role][-2:]
        picked.sort(key=lambda it: it["session"])
        t = router.ask("small", DECIDE + pre + "Most recent evidence per role (oldest first):\n" + bullets(picked),
                       120, usage, ["retrieval-small"])
        p["retrieval-small"] = parse_p(t, log)
        if with_summary:
            t = router.ask("small", DECIDE + pre + "Summary of the evidence:\n" + summary, 120, usage,
                           ["summary-small"])
            p["summary-small"] = parse_p(t, log)
        p["state-code"] = state.p_approve()
        t = router.ask("ideal", DECIDE + state.describe(s + 1), 120, usage, ["orchestrated"])
        p["orchestrated"] = parse_p(t, log)
        # zero-token controls
        p["rule-code"] = truth
        p["additive-code"] = truth_additive(seen)[0]
        p["state-exact-code"] = exact.p_approve()
        out[s + 1] = dict(truth=truth, margin=margin, p=p)
    return out


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def exact_sign_test(b, c):
    """Two-sided exact binomial p-value for b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(min(b, c), n, 0.5).pvalue)
    except Exception:
        from math import comb
        k = min(b, c)
        return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))


def summarise(rows, schemes, usage, P):
    res = {}
    base = {r["project"]: r["correct"] for r in rows if r["scheme"] == BASELINE}
    for s in schemes:
        rr = [r for r in rows if r["scheme"] == s]
        n = len(rr)
        k = sum(r["correct"] for r in rr)
        lo, hi = binom_ci(k, n)
        wrong = [abs(r["p"] - 0.5) * 2 for r in rr if not r["correct"]]
        med = float(np.median([r["margin"] for r in rr]))
        clear = [r["correct"] for r in rr if r["margin"] >= med]
        close = [r["correct"] for r in rr if r["margin"] < med]
        d = dict(n=n, accuracy=k / n, accuracy_ci95=[lo, hi],
                 brier=float(np.mean([(r["p"] - r["truth"]) ** 2 for r in rr])),
                 confidence_when_wrong=float(np.mean(wrong)) if wrong else None,
                 accuracy_clear=float(np.mean(clear)) if clear else None,
                 accuracy_close=float(np.mean(close)) if close else None,
                 share_approved=float(np.mean([r["p"] >= 0.5 for r in rr])))
        if s != BASELINE and base:
            b = sum(1 for r in rr if r["correct"] and not base[r["project"]])      # scheme right, baseline wrong
            c = sum(1 for r in rr if (not r["correct"]) and base[r["project"]])    # baseline right, scheme wrong
            d.update(vs_baseline_scheme_only_right=b, vs_baseline_baseline_only_right=c,
                     vs_baseline_p_value=exact_sign_test(b, c))
        res[s] = d
    return res


def usage_summary(usage, schemes):
    res = {}
    for s in schemes:
        u = usage[s]
        res[s] = dict(calls=u.calls, input_tokens=u.inp, output_tokens=u.out, total_tokens=u.tokens(),
                      output_tokens_per_call=(u.out / u.calls) if u.calls else None, cost_usd=u.cost)
    b = res.get(BASELINE, {}).get("cost_usd") or 0
    for s in schemes:
        res[s]["cost_vs_baseline"] = (res[s]["cost_usd"] / b) if b else None
    return res


# --------------------------------------------------------------------------- #
def main():
    def extra(p):
        p.add_argument("--backend", choices=["mock", "anthropic"], default="mock")
        p.add_argument("--small-model", default=os.environ.get("QDS_SMALL_MODEL", "claude-haiku-5-5"))
        p.add_argument("--ideal-model", default=os.environ.get("QDS_IDEAL_MODEL", "claude-sonnet-5-5"))
        p.add_argument("--price-small", default="1,5", help="USD per million tokens 'input,output'. PLACEHOLDER")
        p.add_argument("--price-ideal", default="3,15", help="USD per million tokens 'input,output'. PLACEHOLDER")
        p.add_argument("--projects", type=int, default=None, help="default 30 (5 with --quick); use 100+ for a claim")
        p.add_argument("--sessions", type=int, default=10)
        p.add_argument("--per-session", type=int, default=4)
        p.add_argument("--checkpoints", default="", help="comma list of session counts to score, e.g. 10,30 (default: last)")
        p.add_argument("--truth", choices=["additive", "order", "both"], default="both")
        p.add_argument("--with-summary", action="store_true", help="add the rolling-summary baseline")
        p.add_argument("--ideal-thinking", choices=["off", "default"], default="off",
                       help="off: ask the ideal model not to think (recorded; some models ignore it)")
        p.add_argument("--half-life", type=float, default=1e9)
        p.add_argument("--state-scale", type=float, default=1.0, help="keep 1.0 unless fixed in advance")
        p.add_argument("--max-calls", type=int, default=6000)
        p.add_argument("--yes", action="store_true")
    args = parse_args(__doc__.splitlines()[0], extra)

    P = args.projects or (5 if args.quick else 30)
    S = args.sessions
    cps = sorted({int(x) for x in args.checkpoints.split(",") if x.strip()} | ({S} if not args.checkpoints.strip() else set()))
    if any(c < 1 or c > S for c in cps):
        sys.exit(f"--checkpoints must lie in 1..{S}")
    modes = ["additive", "order"] if args.truth == "both" else [args.truth]
    schemes = LLM_SCHEMES + (["summary-small"] if args.with_summary else []) + CODE_SCHEMES
    llm_schemes = [s for s in schemes if s not in CODE_SCHEMES]
    prices = {"small": tuple(float(x) for x in args.price_small.split(",")),
              "ideal": tuple(float(x) for x in args.price_ideal.split(","))}
    per_project = S + (S if args.with_summary else 0) + len(cps) * (4 + (1 if args.with_summary else 0))
    n_calls = per_project * P * len(modes)
    d = out_dir(args, "e11_accuracy_eval")
    if args.state_scale != 1.0:
        print(f"WARNING: --state-scale {args.state_scale}: post-hoc if chosen after seeing results; report as exploratory.")
    if n_calls > args.max_calls:
        sys.exit(f"Plan needs {n_calls} calls but --max-calls is {args.max_calls}. Raise it or shrink the run.")

    if args.backend == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Set ANTHROPIC_API_KEY first.")
        print(f"Planned: {n_calls} API calls = {P} projects x {len(modes)} truth(s) x {per_project} calls; "
              f"checkpoints {cps}; small={args.small_model} ideal={args.ideal_model}; thinking for ideal: {args.ideal_thinking}.")
        if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
            sys.exit("Cancelled.")
        small = AnthropicBackend(args.small_model, args.max_calls)
        ideal = AnthropicBackend(args.ideal_model, args.max_calls)
        if args.ideal_thinking == "default":
            ideal.use_no_thinking = False
    else:
        mrng = np.random.default_rng(args.seed + 999)
        small, ideal = MockBackend(mrng), MockBackend(mrng)
        print(f"Mock backend: token counts are ESTIMATES ({n_calls} simulated calls); the 'LLM' is a heuristic, "
              "so accuracy numbers from the mock only test the pipeline.")
    router = Router(small, ideal, prices, args.max_calls)

    all_rows, summary, usage_all, log = [], {}, {}, []
    t0 = time.time()
    for mi, mode in enumerate(modes):
        rng = np.random.default_rng(args.seed + 17 * mi)
        usage = {s: SchemeUsage() for s in llm_schemes}
        rows = []
        for proj in range(P):
            want = float(proj % 2)                                 # alternate -> exactly balanced finals
            items = balanced_items(rng, S, args.per_session, want, TRUTHS[mode])
            res = run_project(router, items, S, args, mode, usage, log, set(cps), args.with_summary)
            for cp, r in res.items():
                for s in schemes:
                    p = float(r["p"][s])
                    rows.append(dict(truth_mode=mode, checkpoint=cp, project=proj, scheme=s, truth=r["truth"],
                                     margin=r["margin"], p=p, correct=int((p >= 0.5) == bool(r["truth"]))))
            print(f"[{mode}] project {proj + 1}/{P} done ({time.time() - t0:.0f}s)")
        all_rows += rows
        summary[mode] = {}
        for cp in cps:
            rr = [r for r in rows if r["checkpoint"] == cp]
            summary[mode][str(cp)] = summarise(rr, schemes, usage, P)
            summary[mode][str(cp)]["_class_balance_approve"] = float(np.mean([r["truth"] for r in rr if r["scheme"] == BASELINE]))
        usage_all[mode] = usage_summary(usage, llm_schemes)

    valid = (len(log) == 0 and small.truncated + ideal.truncated == 0 and small.empty + ideal.empty == 0)
    meta = dict(backend=args.backend, projects=P, sessions=S, checkpoints=cps, truth_modes=modes,
                items_per_session=args.per_session, small_model=args.small_model if args.backend == "anthropic" else "mock",
                ideal_model=args.ideal_model if args.backend == "anthropic" else "mock",
                ideal_thinking_requested=args.ideal_thinking,
                thinking_disabled_in_effect=dict(small=getattr(small, "use_no_thinking", None),
                                                 ideal=getattr(ideal, "use_no_thinking", None)),
                token_counts="estimated" if small.estimated else "reported by API",
                prices_usd_per_mtok=prices, prices_are_placeholders=(args.price_small == "1,5" and args.price_ideal == "3,15"),
                decay=DECAY, state_scale=args.state_scale, half_life=args.half_life, seed=args.seed,
                parse_errors=len(log), truncated=small.truncated + ideal.truncated, empty=small.empty + ideal.empty,
                valid=valid, ci_half_width_at_p_half=1.96 * (0.25 / P) ** 0.5)
    save_json(os.path.join(d, "summary.json"), dict(meta=meta, accuracy=summary, usage=usage_all, parse_errors=log[:20]))
    save_csv(os.path.join(d, "per_project.csv"), all_rows)

    # ---- console report ----------------------------------------------------
    print("\n" + "=" * 100)
    print(f"{'VALID' if valid else 'INVALID (parse errors / truncation: do not report)'}  | n = {P} projects per cell | "
          f"95% interval half-width near p=0.5 is +/-{100 * meta['ci_half_width_at_p_half']:.0f} points")
    if meta["prices_are_placeholders"]:
        print("Prices are PLACEHOLDERS.")
    for mode in modes:
        for cp in cps:
            tab = summary[mode][str(cp)]
            print(f"\n--- truth: {mode} | {cp} sessions | share of 'approve' truths {tab['_class_balance_approve']:.2f}")
            print(f"{'scheme':18s} {'acc':>5s} {'95% CI':>13s} {'Brier':>6s} {'conf|wrong':>10s} {'clear':>6s} {'close':>6s} {'p vs base':>10s}")
            for s in schemes:
                x = tab[s]
                pv = f"{x['vs_baseline_p_value']:.3f}" if "vs_baseline_p_value" in x else "-"
                cw = f"{x['confidence_when_wrong']:.2f}" if x["confidence_when_wrong"] is not None else "-"
                print(f"{s:18s} {x['accuracy']:5.2f} [{x['accuracy_ci95'][0]:.2f},{x['accuracy_ci95'][1]:.2f}] "
                      f"{x['brier']:6.3f} {cw:>10s} {x['accuracy_clear'] if x['accuracy_clear'] is not None else float('nan'):6.2f} "
                      f"{x['accuracy_close'] if x['accuracy_close'] is not None else float('nan'):6.2f} {pv:>10s}")
        u = usage_all[mode]
        print(f"\n--- cost and OUTPUT tokens per call, truth: {mode} (whole run, all checkpoints)")
        print(f"{'scheme':18s} {'calls':>6s} {'out tok/call':>13s} {'cost $':>9s} {'vs base':>8s}")
        for s in llm_schemes:
            x = u[s]
            print(f"{s:18s} {x['calls']:6d} {x['output_tokens_per_call']:13.1f} {x['cost_usd']:9.4f} "
                  f"{(x['cost_vs_baseline'] or 0):8.2f}")
    print("\nHow to read this:")
    print(" * A scheme is 'as accurate' only if its interval overlaps the baseline AND the paired p-value is large AND n is big enough to see a difference you care about.")
    print(" * state-exact-code vs state-code: the gap is extraction error; the distance from rule-code is the state's functional form.")
    print(" * additive-code under the 'order' truth shows what an order-blind memory can reach; the state helps only if it beats that.")
    print(" * Compare out tok/call across schemes before crediting a cost saving to the memory design.")

    # ---- figure ------------------------------------------------------------
    cells = [(m, c) for m in modes for c in cps]
    fig, axes = plt.subplots(1, len(cells), figsize=(4.2 * len(cells), 3.6), squeeze=False)
    for ax, (m, c) in zip(axes[0], cells):
        tab = summary[m][str(c)]
        xs = np.arange(len(schemes))
        acc = [tab[s]["accuracy"] for s in schemes]
        err = [[max(0.0, tab[s]["accuracy"] - tab[s]["accuracy_ci95"][0]) for s in schemes],
               [max(0.0, tab[s]["accuracy_ci95"][1] - tab[s]["accuracy"]) for s in schemes]]
        ax.bar(xs, acc, yerr=err, capsize=2, color=["#555" if s in CODE_SCHEMES else "#2a7f7f" for s in schemes])
        ax.axhline(0.5, ls=":", c="k", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(schemes, rotation=60, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"truth: {m}, {c} sessions (n={P})")
        ax.set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "accuracy.png"))
    print(f"\nWrote {d}/summary.json, per_project.csv, accuracy.png")


if __name__ == "__main__":
    main()
