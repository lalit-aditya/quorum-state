# Evaluation of the results so far (as of 2026-11-07)

This report states what the data supports and what it does not. Every number comes from a file in
`qds_sim/results/`. All dollar figures use **placeholder prices** ($1 / $5 small model, $3 / $15 ideal model
per million input / output tokens). "Saving" is cost relative to the baseline named in each column.

## 1. Short answer for the goal "more than 50% saving"

* Against a **large model re-reading the whole transcript**, the framework's cost is 52–93% lower in every
  real-API run.
* Most of that comes from **using a small model**, not from the framework: the small model re-reading the whole
  transcript is 87–90% cheaper than the baseline in the same runs.
* Against the **small-model baseline** the framework never reached a 50% saving. It cost more at 10 sessions,
  saved 42% at 30 sessions (state, code only), and cost far more when answers are needed only every 5th session.
* **Accuracy fell** in every real-API run (state schemes 0.4–0.6 against 0.9–1.0 for the large model re-reading
  the transcript). With 5–10 projects the intervals overlap, so this is not proven, but all point estimates point
  the same way.
* About 64–74% of the baseline's cost is **output tokens** (266–468 per decision although the reply is capped at
  120), against 61–82 output tokens per decision for the ideal model fed only the state. If output length were
  equal, the orchestrated saving would be about 14% (10 sessions), 52% (30 sessions) and 0% (every 5th session).
  Whether the extra output is hidden reasoning is not yet established (see `thinking_disabled_in_effect`, which
  `e11_accuracy_eval.py` records).

## 2. Real-API orchestration runs (e8)

Models: `claude-haiku-5-5` (small), `claude-sonnet-5-5` (ideal). Token counts are reported by the API.
All runs: 0 parse errors, 0 truncated replies, 0 empty replies.

| Run | Projects | Calls | Accuracy: ideal / small-transcript / state / orchestrated | Saving vs large-transcript: state / orchestrated / small-transcript |
|---|---|---|---|---|
| `e8_s10` (10 sessions, answer every session) | 10 | 500 | 0.9 / 0.8 / 0.6 / 0.6 | 87% / 63% / 90% |
| `e8_s30` (30 sessions, answer every session) | 5 | 750 | 1.0 / 0.4 / 0.6 / 0.6 | 93% / 77% / 87% |
| `e8_s30_d5` (30 sessions, answer every 5th) | 5 | 270 | 1.0 / 0.4 / 0.4 / 0.4 | 67% / 52% / 87% |

Against the small-transcript baseline: state saves −34% / 42% / −162% and orchestrated −269% / −84% / −278%
(negative = costs more).

| Run | Output tokens per decision: baseline (large model) | Ideal model fed only the state | Baseline output share of cost |
|---|---|---|---|
| `e8_s10` | 266 | 61 | 74% |
| `e8_s30` | 420 | 79 | 64% |
| `e8_s30_d5` | 468 | 82 | 64% |

## 3. Token benchmark (e7)

* The paper's e7 table (15 projects at 10 sessions, 15 at 30) reports state memory using 38% fewer tokens than the
  transcript at 10 sessions and 74% fewer at 30, with accuracy 0.47 against 0.73 at 10 sessions.
* The replicate in `results/e7_s10` (10 projects) agrees: state = 0.62 × transcript tokens, accuracy 0.60 against
  0.80, intervals overlapping.
* The 30-session replicate **timed out and wrote no results** (`results/e7_s30.log`, `RESULTS_e7.md`).
* Retrieval was cheaper than the state at every history length, and less accurate.

## 4. Why the state loses accuracy (e10, no API)

The benchmark's ground truth is an additive rule. The state's rotations do not commute, so even with perfect
extraction it reproduces that rule only 69% of the time at 10 sessions and 48% at 30 (rotation scale 1.0). At
scale 0.1 it reproduces it about 89–90%, but then order effects disappear, which removes the framework's
distinguishing property. The state is also more sensitive to small extraction noise than the additive rule.
The accuracy gap in the API runs is therefore mostly the state's functional form, not LLM extraction noise.

## 5. Hierarchy (e9)

Only a 3-project, 5-session real-API pilot exists (`results/e9_pilot`, 66 calls); `results/e9_hierarchy` is a
mock run. The ground truth is a synthetic rule. No claim about the hierarchy is supported yet.

## 6. Simulation studies (e0–e6)

Experiment 0 uses real published data (Wang & Busemeyer 2013). Experiments 1–6 generate data from the models, so
they test the analyses and give effect sizes and sample sizes; they are not evidence about people or LLMs.

## 7. What would change the picture

1. `e11_accuracy_eval.py` with 60+ projects: does the state beat the order-blind control (`additive-code`) on the
   order-dependent truth, and is the gap to the baseline real at adequate n?
2. A cached-prefix baseline and a rolling-summary baseline in the cost comparison.
3. Equal output caps and a recorded thinking setting for baseline and orchestrated schemes.
4. A real decision rule and real prices in place of the synthetic rule and placeholder prices.
