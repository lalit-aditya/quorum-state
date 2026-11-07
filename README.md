# Order, Timing, and Feedback: a quantum-probability decision state for role-aware agents

Preprint source, simulation package, test suite, raw results and reports for the project by
**Lalit Aditya R D** (VIT; Zelleo). Quantum probability is used purely as mathematics on ordinary computers;
no quantum hardware is involved.

> **Status in one paragraph.** The theory and the simulations reproduce and are tested (24 automated checks).
> The token and cost experiments with real LLM calls show large cost reductions against a large model that
> re-reads the whole transcript, but most of that saving comes from using a small model, and accuracy was lower
> for the decision-state schemes in every real-API run (small samples, overlapping intervals). The strongest
> open question is whether the state is *more* accurate than order-blind memories on tasks where evidence order
> matters. `qds_sim/experiments/e11_accuracy_eval.py` was written to answer it and has **not yet been run with the
> real API**. Read `reports/EVALUATION.md` before quoting any saving figure.

---

## 1. The idea in plain language

An AI agent that makes decisions for several stakeholders (finance, engineering, operations) usually keeps a
long transcript and re-reads it every time. This project replaces the transcript with a **decision state**: a
single two-level quantum-probability state (a qubit density matrix).

* **Evidence** (a sentence like "engineering reports the schedule slipped") is read once by a small model into
  three numbers: *role*, *direction* (+1 supports approving, −1 does not), *strength* in [0, 1].
* Each item **rotates** the state about a role-specific axis. Roles use different, non-commuting axes, so the
  **order** of evidence changes the outcome, as order effects do in human judgement.
* **P(approve) = Tr(P ρ)** is read from the state in code. The state is a few numbers and does not grow with history.
* **Timing:** a review (a Lüders projection, optionally unsharp) and drift toward the default decision ("do not
  approve") model repeated reviews and reminders. The review schedule acts as a spectral filter (an *overlap law*),
  giving decision-Zeno and anti-Zeno regimes.
* **Feedback:** a linear-Gaussian surrogate has an interior optimum gain, so coordination between roles can be
  tuned rather than maximised.
* **Orchestration:** a small model extracts, code updates the state, and a larger ("ideal") model answers from a
  one-line description of the state. The paper's framework figure is in `docs/framework_figure.png`.

The claim being tested is not "this is more accurate than an LLM". It is: *can this state keep a long decision
history cheaply, show human-like order effects, and be controlled through timing and feedback?*

## 2. Repository map

```
README.md                         this file
paper/
  main.tex, refs.bib, main.bbl    LaTeX source of the preprint (26 pages), figures in paper/figs
  Quantum_Decision_State_preprint_filled.pdf   compiled paper with all results filled in
  code/                           three small helper scripts that came with the paper source (QQ and simulation figures)
docs/
  framework_figure.png / .pdf     step-by-step framework figure (also in the paper)
reports/
  EVALUATION.md                   honest evaluation of every result against the 50%-saving goal
  RESULTS_e7.md                   notes on the e7 real-API token runs, including the failed 30-session run
  CLAUDE_CODE_INSTRUCTIONS.md     paste-ready instructions for running further experiments in Claude Code
qds_sim/                          the Python package and experiments
  qds/                            core library (states, rotations, review, drift, feedback, order-effect models)
  experiments/                    one script per study, e0 to e11 (+ replot_e11.py)
  tests/                          4 test files, 24 checks
  results/                        every output folder produced so far (CSV, JSON, PNG, logs)
  RUN_E8.md, RUN_E9.md, RUN_E11.md   exact commands for the API experiments
  CLAUDE.md                       rules for Claude Code when it runs the benchmark
  run_all.py                      tests + every experiment (mock backends, no API cost)
```

## 3. Quick start (no API, no cost)

```bash
cd qds_sim
pip install -r requirements.txt          # Python 3.10+
python tests/test_core.py                # 13 checks: the paper's propositions and published numbers
python tests/test_pipeline.py            # 4 checks: step-by-step pipeline, noise operator
python tests/test_hierarchy.py           # 3 checks: macro/meso/micro hierarchy
python tests/test_e11.py                 # 4 checks: accuracy-evaluation script
python run_all.py --quick                # all experiments with mock backends (about 1.5 minutes)
```

Each experiment can be run alone from `qds_sim/experiments/`, e.g. `python e1_zeno_schedule.py --seed 1`. Output
goes to `qds_sim/results/<name>/`. Fixed seeds make every simulation reproducible; a rerun of e10 by the author
matched the paper's Table 5 exactly.

## 4. The experiments

| Script | What it tests | Data |
|---|---|---|
| `e0_reanalysis.py` | Re-analysis of published order-effect data (paper Table 2) | **Real** (Wang & Busemeyer 2013) |
| `e1_zeno_schedule.py` | H1 decision Zeno; H2 anti-Zeno and the white-noise null | Synthetic |
| `e2_reminders.py` | H3 reminders versus recommitment | Synthetic |
| `e3_delay_qq.py` | H4 delay breaks the QQ equality; power | Synthetic |
| `e4_feedback.py` | H5 interior gain optimum, coordination death | Synthetic |
| `e5_model_recovery.py` | Can a study tell the models apart? (symmetric classical anchoring satisfies QQ exactly) | Synthetic |
| `e6_agent_memory.py` | Study-3 harness: transcript memory versus quantum memory | Synthetic, mock LLM |
| `e7_token_benchmark.py` | Tokens and accuracy of transcript / summary / retrieval / state memories | Real API or mock |
| `e8_orchestrated.py` | Step-by-step framework with a small extractor and an ideal answerer, tokens and dollars | Real API or mock |
| `e9_hierarchy.py` | Macro / meso / micro hierarchy with uncertainty-gated escalation | Real API or mock; **synthetic ground truth** |
| `e10_noise_stability.py` | Noise stability of the state (adapts Haris et al. 2026); fidelity to an additive rule | Simulation, no API |
| `e11_accuracy_eval.py` | Accuracy of every scheme, balanced truths, additive and **order-dependent** truths, paired tests, zero-token controls, output tokens per call | Real API or mock |

Synthetic is not evidence about people or LLMs. Experiments 1–6 generate data from the models themselves; they
check that the analyses work and estimate effect and sample sizes. Only e0 uses real human data.

## 5. Results at a glance

All dollar figures use **placeholder prices** ($1/$5 small, $3/$15 ideal model per million tokens). Details and the
full tables are in `reports/EVALUATION.md`.

**Real-API runs (small model `claude-haiku-5-5`, ideal model `claude-sonnet-5-5`):**

| Run | Accuracy: large-transcript / small-transcript / state / orchestrated | Cost saving vs large-transcript: state / orchestrated / small-transcript |
|---|---|---|
| 10 sessions, 10 projects, answer every session | 0.9 / 0.8 / 0.6 / 0.6 | 87% / 63% / 90% |
| 30 sessions, 5 projects, answer every session | 1.0 / 0.4 / 0.6 / 0.6 | 93% / 77% / 87% |
| 30 sessions, 5 projects, answer every 5th session | 1.0 / 0.4 / 0.4 / 0.4 | 67% / 52% / 87% |

What this means:

1. A saving above 50% against a large model that re-reads the transcript is observed in all three runs.
2. Switching to the small model alone saves 87–90%, so the framework's own contribution is not established.
   Against the small-model baseline the framework cost more at 10 sessions, saved 42% at 30 sessions (state only),
   and cost far more when answers are needed every 5th session.
3. 64–74% of the large-model baseline's cost is output tokens (266–468 per decision, against 61–82 for the ideal
   model fed only the state). At equal output length the orchestrated saving would be about 14%, 52% and 0%.
4. The state schemes were less accurate in every run. n = 5–10 projects, so intervals overlap.
5. The cause is mostly the state's functional form: with perfect extraction it reproduces an additive rule only 69%
   (10 sessions) and 48% (30 sessions) of the time (e10). The benchmark's additive truth therefore favours
   additive memories; an order-dependent truth is the fair test (e11).

**Simulation highlights (synthetic):** P(guilty) falls 0.47 → 0.09 with more judgments (H1); reminders beat
recommitment when the environment is slow (H3); interior optimum gain 1.14 / 0.76 / 0.43 for η = 1 / 0.5 / 0.2 (H5).
**Real data (e0):** the QQ-constrained model has the best AIC in 4 of 6 data sets; rank-one models are rejected in all.

## 6. Which result folder is what

| Folder in `qds_sim/results/` | Kind | Status |
|---|---|---|
| `e0_reanalysis` … `e6_agent_memory` | Re-analysis and simulations | Complete |
| `e7_mock` | Mock backend, estimated tokens | Pipeline check only |
| `e7_pilot` | Real API, 1 project × 3 sessions (18 calls) | Valid, no statistical information |
| `e7_s10` | Real API, 10 projects × 10 sessions (600 calls) | Valid |
| `e7_s30.log` | Real API, 30 sessions | **Failed** (API timeout), no results |
| `e7_ckpt_test`, `e7_ckpt_crash` | Checkpoint / resume tests | Test artefacts |
| `e8_orchestrated` | Mock | Pipeline check only |
| `e8_pilot` | Real API, 3 × 5 (75 calls) | Valid, pilot |
| `e8_s10`, `e8_s30`, `e8_s30_d5` | Real API (500, 750, 270 calls) | Valid; see section 5 |
| `e9_hierarchy` | Mock | Pipeline check only |
| `e9_pilot` | Real API, 3 × 5 (66 calls), synthetic truth | Valid pilot; no claim supported |
| `e10_noise_stability` | Simulation, no API | Complete, reproduced independently |
| `e11_accuracy_eval` | — | **Not yet run with the real API** |

A real run is only valid with 0 parse errors, 0 truncated and 0 empty replies; `summary.json` records this.

## 7. Running the API experiments safely

* Set the key in the environment only; never put it in a file, a log or a chat:
  `export ANTHROPIC_API_KEY=...` (PowerShell: `$env:ANTHROPIC_API_KEY="..."`).
* Every API script prints the planned number of calls and asks before spending; `--max-calls` is a hard cap.
* **Set your real prices** with `--price-small IN,OUT --price-ideal IN,OUT`; the defaults are placeholders and
  `summary.json` records whether they were used.
* Do not change prompts, ground truth or `--state-scale` after seeing results; use fresh `--seed` values for
  confirmation runs. Post-hoc tuning must be reported as exploratory.
* The next run to do: `RUN_E11.md` (about 8,400 calls for 60 projects, both truths, with the rolling-summary baseline).

## 8. What is and is not supported

| Claim | Status |
|---|---|
| Order effects arise from non-commuting role rotations; QQ holds for the model; symmetric classical carry-over also satisfies QQ | Proved and tested (`test_core.py`) |
| Review schedule acts as a spectral filter (Zeno / anti-Zeno); white-noise null | Supported in simulation |
| Interior optimal feedback gain exists in the surrogate | Supported in simulation |
| State memory uses fewer tokens than re-reading a transcript | Supported in templated benchmarks; retrieval is cheaper still |
| Framework saves more than 50% **against a small-model or summary baseline** | **Not supported** by current data |
| State is as accurate as transcript memory | **Not supported**; point estimates lower, n small |
| State beats order-blind memory when evidence order matters | **Untested** (e11) |
| Macro/meso/micro hierarchy | **Untested** beyond a 66-call pilot |
| H1–H5 in humans | **Untested** (no human studies yet) |

## 9. Known open items

* Paper placeholders: repository link and acknowledgements are still "to be completed".
* References [28]–[30] are marked "to be verified" while the AI-use statement says all references were verified;
  resolve one or the other before submission.
* The paper is single-author in its latest version; restore co-authors if wanted.
* Prices in all cost tables are placeholders; replace with actual prices.
* The hierarchy's ground truth and the meso plan rule are synthetic stand-ins.
* Missing baselines in the cost comparison: cached-prefix (prompt caching) is not implemented; the rolling summary
  exists in e11 (`--with-summary`) and e7.
* Larger runs (50+ projects, balanced truths, ideal-model thinking setting recorded) are not done.

## 10. Correction log

* Agent-memory role axes: an earlier version used the x, y, z axes, so engineering and operations evidence initially
  had no effect on P(approve). All role axes now have a y-component.
* Paper Table 2, rank-one column: reverse-order cells were listed in the wrong order in an earlier script; fixed.
* e7–e9: a comment claimed the state "matches the ground truth". It does not (see e10); the gap in the API runs is
  mostly the state's functional form, not extraction noise. Comments and README corrected.
* Paper Section 7.4: g* corrected to 1.14 / 0.76 / 0.43 and the coordination-death time to 1.61.
* e10: the noise operator now uses keep probability ρ so ρ is the item correlation; documented in the paper.
* Earlier statement that the state's accuracy gap was mainly extraction noise was wrong; e10 showed the functional form
  is the larger limit.

## 11. Citation and licence

Cite as: Lalit Aditya R D, *Order, Timing, and Feedback: A Quantum-Probability Decision State for Role-Aware Agents*
(preprint, 2026). The noise-stability analysis adapts Haris, Zhang and Yoshida (ICLR 2026, arXiv:2602.08287); the
overlap law follows Kofman and Kurizki (2005); the real data are from Wang and Busemeyer (2013).
**No licence has been chosen yet.** Add one before making the repository public.
