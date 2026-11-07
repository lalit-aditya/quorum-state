# E7 — Token benchmark results

Experiment: `experiments/e7_token_benchmark.py`. Question: does the decision-state
memory use fewer tokens than transcript, summary and retrieval memories at
comparable decision quality?

- **Model:** `claude-haiku-5-5` (all runs)
- **Date:** 2026-11-07
- **Token counts:** reported by the API (`usage`), not estimated
- **Extraction:** batched per session (default; `--per-item-extract` not run)

## Runs

| run | output folder | projects | sessions | planned calls | status |
|---|---|---|---|---|---|
| mock check | `results/e7_mock` | 3 | 10 | 0 (offline, estimated tokens) | completed |
| pilot | `results/e7_pilot` | 1 | 3 | 18 | completed |
| medium history | `results/e7_s10` | 10 | 10 | 600 | completed (672 s) |
| long history | `results/e7_s30` | 5 | 30 | 900 | **FAILED** — see "Failures" |

Total real API calls actually issued: 18 (pilot) + 600 (s10) + roughly 540–700
(s30, aborted during project 4 of 5; the exact count was not written out because
the run crashed before producing any output files).

## Pilot — 1 project x 3 sessions (18 calls)

```
scheme       calls    tokens vs transcript  accuracy  Brier  agree
transcript       3       732          1.00      0.00  0.303   1.00
summary          6      1873          2.56      1.00  0.250   0.00
retrieval        3       600          0.82      1.00  0.160   0.00
state            3       920          1.26      1.00  0.007   0.00
state+llm        6      1328          1.81      1.00  0.006   0.00
```

Purpose was pipeline validation only: 0 parse errors, 0 truncated, 0 empty
replies, token counts reported by the API. Accuracy here is one project, so it
carries no information (95% interval spans the whole [0, 1] range). At three
sessions the state scheme is *more* expensive than transcript (1.26x) because the
fixed per-session extraction cost has nothing to amortise against yet.

## Medium history — 10 projects x 10 sessions (600 calls)

```
scheme       calls    tokens vs transcript  accuracy  Brier  agree
transcript     100     48727          1.00      0.80  0.218   1.00
summary        200     76644          1.57      0.50  0.282   0.70
retrieval      100     20380          0.42      0.40  0.332   0.60
state          100     30420          0.62      0.60  0.353   0.60
state+llm      200     44155          0.91      0.60  0.289   0.60
```

Accuracy is scored once per project (on the final decision), so n = 10. With the
95% interval p +/- 1.96*sqrt(p(1-p)/n):

| scheme | accuracy | 95% interval |
|---|---|---|
| transcript | 0.80 | 0.80 +/- 0.25 |
| summary | 0.50 | 0.50 +/- 0.31 |
| retrieval | 0.40 | 0.40 +/- 0.30 |
| state | 0.60 | 0.60 +/- 0.30 |
| state+llm | 0.60 | 0.60 +/- 0.30 |

**All of these intervals overlap.** At n = 10 this run cannot distinguish any
scheme from any other on accuracy.

## How the state token ratio changes from 10 to 30 sessions

**Not measured.** The 30-session run failed before writing results, so the key
quantity this benchmark was built to measure — whether the state's token
advantage grows with history length — has no real-API number. What exists:

- 3 sessions (pilot, n = 1 project): state = 1.26x transcript
- 10 sessions (s10): state = 0.62x transcript

The direction is consistent with the expected amortisation (transcript grows
linearly with history, the state does not), but two points, one of which is a
single project, do not establish a trend. The 30-session run must be repeated.

## Does `state` match `transcript`, and how does it compare with `retrieval`?

On this evidence, **no — and the comparison is not resolvable at this sample
size.**

- `state` uses 0.62x the tokens of `transcript` and scored lower on accuracy
  (0.60 vs 0.80) and worse on Brier (0.353 vs 0.218). The accuracy gap is well
  inside noise (intervals overlap heavily), so this is *not* evidence that
  `state` is worse — it is an absence of evidence that it matches.
- `retrieval` remains the cheapest scheme (0.42x) as expected, and scored lowest
  on accuracy (0.40) and second-worst on Brier (0.332). `state` costs about 1.5x
  what `retrieval` costs for a 0.20 accuracy difference that is not
  statistically distinguishable at n = 10.
- `summary` is the clear loser on cost: 1.57x transcript tokens (it pays for both
  the summarisation call and the decision call) at 0.50 accuracy.
- Agreement with transcript: `state`, `state+llm` and `retrieval` all at 0.60,
  `summary` at 0.70.

Stated plainly: **this run does not support the claim that the decision state
keeps decision quality close to transcript at lower cost.** It establishes the
token saving (0.62x, which is real and measured by the API) but not the quality
parity. Per the project's own standard — tokens alone are not a win — the
benchmark is currently inconclusive, leaning negative for `state` against both
`transcript` (quality) and `retrieval` (cost).

## Parse errors and failures

- Mock, pilot and s10: **0 parse errors, 0 truncated replies, 0 empty replies.**
  `summary.json` reports `"valid": true` for s10.
- Both API runs logged `note: temperature not accepted; continuing without it`.
  The run continues with the model default rather than `temperature=0`, so the
  decisions are not fully deterministic.
- **s30 crashed**, exit code 1, after completing 3 of 5 projects (618 s elapsed):

  ```
  anthropic.APITimeoutError: Request timed out or interrupted. This could be due to
  a network timeout, dropped connection, or request cancellation.
  ```

  The SDK client is constructed with `max_retries=6`, and the retry loop in
  `ask()` only re-tries requests rejected for `temperature`/`thinking`, so a
  transport timeout that survives the SDK's own retries propagates and kills the
  whole run. No output folder was written: results for the three completed
  projects were lost. The run was issued in parallel with the 600-call s10 run
  against the same API key, which is the most likely contributing cause.

  There is no resume or checkpoint facility, so a repeat run re-spends the full
  900 calls.

## Limitations

- **Synthetic templated evidence.** Evidence sentences are generated from
  templates, not drawn from real project histories; absolute accuracy numbers do
  not transfer to real workloads.
- **The ground truth favours additive memories.** Approval is defined as the
  equal-weight sum of signed evidence strengths being positive. Any scheme that
  accumulates evidence additively is structurally advantaged; order-dependent or
  overriding evidence is not tested here.
- **Very small samples.** Accuracy is one observation per project: n = 10 for
  s10, n = 1 for the pilot. The 95% intervals above are +/- 0.25 to 0.31 and all
  overlap. No accuracy conclusion is supportable.
- **One model, one configuration.** Only `claude-haiku-5-5`, batched extraction,
  default temperature (temperature=0 was rejected). No second model was run.
- **The long-history condition is missing**, which is the condition under which
  the state scheme is predicted to win.

## What would make this conclusive

1. Re-run the 30-session condition (900 calls), serially rather than alongside
   another run.
2. Raise the project count substantially — n = 10 is far too small for a 0.20
   accuracy difference. Detecting that difference needs on the order of 100
   projects per condition.
3. Add checkpointing or per-project output flushing to `e7_token_benchmark.py` so
   a transport timeout does not discard completed work.
