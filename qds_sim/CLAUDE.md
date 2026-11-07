# Instructions for Claude Code

This repository holds the simulation code for the preprint *Order, Timing, and
Feedback: A Quantum-Probability Decision State for Role-Aware Agents* (author:
Lalit Aditya R D). The current task is the **token benchmark**
(`experiments/e7_token_benchmark.py`): measure, with real API token counts, whether
the decision-state memory uses fewer tokens than transcript, summary and retrieval
memories at comparable decision quality.

## Ground rules

- Never print, log, echo or commit the value of `ANTHROPIC_API_KEY`. Only check that it is set.
- Do not change the ground-truth definition, prompts, or scoring to make any scheme look
  better. If a scheme (including `state`) loses, report that plainly.
- Stop and ask the user before any run that will make more than 100 API calls.
- Keep every run's output folder; do not overwrite earlier results (use `--out`).
- Do not edit the paper. Write findings to `RESULTS_e7.md` only.

## Steps

1. Environment
   ```bash
   python -m pip install numpy scipy matplotlib anthropic
   python tests/test_core.py
   ```
   All tests must pass. Confirm `ANTHROPIC_API_KEY` is set without printing it.

2. Offline check (no API calls)
   ```bash
   cd experiments
   python e7_token_benchmark.py --quick --out ../results/e7_mock
   ```

3. Pilot (18 API calls)
   ```bash
   python e7_token_benchmark.py --backend anthropic --projects 1 --sessions 3 --yes \
       --out ../results/e7_pilot
   ```
   Check `summary.json`: `parse_errors` should be 0 and token counts should say
   "reported by API". If the model name is rejected, list the available models from
   the docs (https://docs.claude.com/en/api/overview) and ask the user which to use
   (set it with `--model` or the `QDS_MODEL` environment variable).

4. Main runs (ask the user before each)
   ```bash
   # 600 calls: medium history
   python e7_token_benchmark.py --backend anthropic --projects 10 --sessions 10 --yes --out ../results/e7_s10
   # 900 calls: long history
   python e7_token_benchmark.py --backend anthropic --projects 5 --sessions 30 --yes --out ../results/e7_s30
   # optional: unbatched extraction, to measure call overhead
   python e7_token_benchmark.py --backend anthropic --projects 5 --sessions 10 --per-item-extract --yes \
       --out ../results/e7_s10_peritem
   ```
   Use `--max-calls` as a hard budget cap. Repeat a main run with a second model if
   the user agrees.

5. Report: write `RESULTS_e7.md` containing
   - model(s), date, number of projects/sessions, total calls;
   - for each run, the table printed at the end (calls, tokens, ratio to transcript,
     accuracy, Brier score, agreement with transcript);
   - how the token ratio of `state` changes from 10 to 30 sessions;
   - whether `state` matches `transcript` accuracy, and how it compares with `retrieval`
     (the cheapest baseline);
   - parse errors and any failures;
   - limitations: synthetic templated evidence; ground truth is an equal-weight sum
     that favours additive memories; small numbers of projects mean wide uncertainty
     (give accuracy with a 95% interval: p +/- 1.96*sqrt(p(1-p)/n)).

## Interpreting results honestly

- Tokens alone are not a win: `retrieval` is usually the cheapest scheme. The decision
  state is worth it only if it keeps decision quality close to `transcript` at lower
  cost, especially as histories get longer.
- Role axes in the quantum state (`ROLES` in `e6_agent_memory.py`) all have a
  y-component, so every role's evidence moves P(approve). An earlier version used the
  x, y and z axes, under which engineering and operations evidence initially had no
  effect; do not revert to that.
