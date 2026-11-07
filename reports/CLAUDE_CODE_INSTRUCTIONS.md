# Claude Code instructions: test the >50% saving claim

Run from the root of the `qds_sim` folder. Paste the prompt in section 1 into Claude Code, then follow sections 2 to 4.

---

## 1. Prompt to paste into Claude Code

```text
Read CLAUDE.md, README.md, experiments/e8_orchestrated.py and experiments/e10_noise_stability.py first.

Goal: test whether the framework can save more than 50% of cost without losing accuracy, against the strongest simple baselines. Do not tune anything after seeing results. Fix every setting below before the first run, and report whatever comes out, including a negative result.

TASK 1: add two baselines to e8_orchestrated.py
- "cached-prefix": the ideal model re-reads the full transcript on every decision, but the repeated prefix is billed at a cache-read price. Make the cache price a CLI flag (--cache-read-frac, default 0.10 of the input price) and print that it is an assumption.
- "rolling-summary": the small model keeps a running text summary, updated after each session. The ideal model answers from that summary only.
- Keep the existing five schemes unchanged. Add the new ones to the same usage and cost accounting.
- Savings must be reported against the cheapest baseline that is still accurate, not only against the full-transcript baseline.

TASK 2: create experiments/e11_order_dependent.py (no API calls)
- Build a synthetic ground truth whose correct answer depends on evidence order. Example: a later evidence item overrides an earlier one only if it comes from a different role, and the same item in the opposite order gives the opposite answer. Document the rule in the docstring, marked "REPLACE with your real rule".
- Compare: the quantum state at rotation scale 1.0 (fixed in advance), the additive rule, the rolling-summary logic and a full-transcript oracle.
- Use 200 projects, sessions 10, 30 and 60, fixed seeds, 95% Wilson intervals and Brier score.
- Output CSVs, summary.json and a figure. Report where the state beats the additive rule and where it does not. If it does not beat it anywhere, say so plainly in the summary.

TASK 3: create experiments/e12_breakeven.py (no API calls)
- Inputs: input and output price per million tokens for the small and ideal model, decisions per project, tokens per session, and history length.
- Output: cost per scheme, saving percent against each baseline, accuracy next to each (read from the e8/e11 results files), and the history length at which saving crosses 50%. Add a --hybrid option that uses the full-transcript baseline below the break-even length and the state scheme above it.

TASK 4: tests and docs
- Add tests for the new schemes and for e11 and e12. Run the whole suite.
- Add e11 and e12 to README.md, and a RUN_E11.md with the exact commands. Add a correction-log entry if you change any earlier claim.
- Add everything to run_all.py.

Rules: do not invent numbers. Mark all prices as placeholders. Do not edit the paper's LaTeX. At the end, print a short table of saving percent and accuracy per scheme at 10, 30 and 60 sessions.
```

---

## 2. Commands to run after Claude Code finishes

```bash
python -m pytest tests -q
python experiments/e11_order_dependent.py
python experiments/e8_orchestrated.py --backend mock --projects 20 --sessions 30 --demo
python experiments/e12_breakeven.py --hybrid
```

---

## 3. Real API run (only after the mock run looks sane)

Use your actual model prices, because the defaults are placeholders. Check the `--price-*` argument format against `RUN_E8.md` before running.

```bash
python experiments/e8_orchestrated.py --backend anthropic \
  --projects 50 --sessions 30 --max-calls 3000 \
  --price-small 1,5 --price-ideal 3,15 --yes
```

---

## 4. What to send back

Send the new `summary.json` files from e8, e11 and e12 so the paper tables can be updated.

---

## 5. How to read the results

- **Saving is only real if accuracy holds.** Compare saving percent and accuracy side by side for every scheme.
- **The baseline that matters is cached-prefix or rolling-summary,** not full-transcript re-reading. If one of them already saves most of the cost, the state's contribution is small.
- **e11 decides whether the state has an accuracy case.** If it does not beat the additive rule on the order-dependent task, the only argument for the state is token saving.
- **At short histories the bill can rise.** Use the `--hybrid` option so the state only switches on past the break-even length.
