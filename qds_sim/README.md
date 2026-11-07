# qds_sim — simulations for *Order, Timing, and Feedback*

Code accompanying the preprint *Order, Timing, and Feedback: A Quantum-Probability
Decision State for Role-Aware Agents*. Quantum probability is used purely as
mathematics on ordinary computers; no quantum hardware is involved.

## Quick start

```bash
pip install numpy scipy matplotlib      # Python 3.10+
python tests/test_core.py               # 13 checks of the paper's propositions
python run_all.py --quick               # every experiment, fast mode (~2 min)
python run_all.py                       # full runs (~7 min)
```

Each script can also be run alone from `experiments/`, e.g.
`python e1_zeno_schedule.py --quick --seed 1`. Results (CSV, JSON, PNG) go to
`results/<script name>/`.

## Layout

| Path | Contents |
|---|---|
| `qds/core.py` | States, rotations, Lüders projection, unsharp review, amplitude damping, sequential probabilities, QQ statistic, commutator (role-conflict) norm |
| `qds/drift.py` | Overlap law R = 2π∫G·F (Kofman & Kurizki 2005), Lorentzian and white-noise environments, arbitrary review schedules |
| `qds/feedback.py` | Two-role linear-Gaussian model, closed-loop Lyapunov covariance, mutual information, coordination-death time, Euler–Maruyama cross-check |
| `qds/models.py` | Order-effect data: q-test, order-free / QQ-constrained / saturated fits, synthetic generators, the six published data sets |
| `tests/test_core.py` | Propositions 1–3, Lemma 1, QQ invariance, published numbers, Monte Carlo agreement |
| `experiments/` | One script per study (below) |

## Experiments

| Script | Tests | Data | Main result (default seeds) |
|---|---|---|---|
| `e0_reanalysis.py` | Paper Table 2 | **Real** (Wang & Busemeyer 2013) | QQ-constrained model best AIC in 4/6; rank-one models rejected everywhere (G² ≥ 26, 2 df) |
| `e1_zeno_schedule.py` | H1 decision Zeno; H2 anti-Zeno and white-noise null | Synthetic | P(guilty) falls 0.47 → 0.09 with more judgments; two-stage prediction error 0.005. Detuned decision: retention 0.50 → 0.23 at τ ≈ τ_c; white noise: no schedule effect |
| `e2_reminders.py` | H3 reminders vs recommitment | Synthetic | Reminders beat recommitment when the environment is slow (0.75 vs 0.57 retention, K = 4); advantage disappears when it fluctuates fast; recommitment leaves larger later order effects |
| `e3_delay_qq.py` | H4 delay breaks QQ; power | Synthetic | Drift alone: median \|q\| 0.08 at strong drift, but model-dependent (≈0 in the example model). Intervening content: q ≈ 0.14, ≥ 90% power with 250 per order up to 1 h delay |
| `e4_feedback.py` | H5 interior gain optimum | Synthetic | Optimal gain 1.15 / 0.75 / 0.45 for η = 1 / 0.5 / 0.2; feedback g ≥ 0.6 prevents coordination death |
| `e5_model_recovery.py` | Can a study tell models apart? | Synthetic | Symmetric classical anchoring satisfies QQ **exactly**; asymmetric anchoring caught only 38% of the time at N = 1000 per order |
| `e7_token_benchmark.py` | Does state memory reduce tokens? | Real API (`--backend anthropic`) or mock | See "Token benchmark" below |
| `e8_orchestrated.py` | Step-by-step framework: small extractor + ideal answerer vs traditional memories, tokens and dollars | Real API (`--backend anthropic`) or mock | `--demo` prints every step; see RUN_E8.md |
| `e9_hierarchy.py` | Macro / meso / micro hierarchy: three level states, small extractor, ideal answerer, uncertainty-gated escalation | Real API or mock; SYNTHETIC ground truth | `--demo` prints every level; see RUN_E9.md |
| `e10_noise_stability.py` | Noise stability of the decision state (adapts Haris et al. 2026) and its fidelity to an additive rule with exact extraction | Simulation, no API | `python e10_noise_stability.py --projects 1000` |
| `e11_accuracy_eval.py` | Accuracy of every memory scheme, balanced truths, additive and ORDER-dependent ground truths, paired tests, zero-token controls, output tokens per call | Real API or mock | See RUN_E11.md; run with the real API for results |
| `e6_agent_memory.py` | Study 3 harness | Synthetic (mock LLM) | Pipeline demo: memory 425 tokens (transcript) vs 4 numbers (quantum); only the quantum memory shows question-order effects |

## Read before reporting results

1. **Synthetic is not evidence.** Experiments 1–6 generate data *from the models*. They
   check that the analyses work and estimate effect sizes and sample sizes. They do
   not show that people or LLMs behave this way. Only `e0` uses real data.
2. **Experiment 6 uses a mock LLM by default**, a template parser with noise. To use
   Claude: `pip install anthropic`, set `ANTHROPIC_API_KEY`, then
   `python e6_agent_memory.py --backend anthropic --model <model-id> --projects 10`.
   The default model id is set by `--model` or the `QDS_MODEL` environment variable;
   check current model names at https://docs.claude.com/en/api/overview. This backend
   has not been tested here (no network access during development), and a run makes
   one API call per evidence sentence (`--projects × --sessions × --per-session`).
3. **Identifiability (e1).** The no-judgment group's P(guilty) = sin²(12δ/2) is
   periodic in the evidence strength δ, so the two-stage fit restricts δ to the
   monotone branch 12δ ≤ π. State this assumption when fitting real data.
4. **H4 is heterogeneous (e3).** The size of a delay-induced QQ violation depends on
   where the default state lies. Include the intervening-content manipulation or
   plan for large samples.
5. **QQ alone is weak evidence (e5).** Simple classical mechanisms can pass the q-test.
   Support for the framework must come from the joint pattern of H1–H5.

## Token benchmark (Experiment 7) with Claude Code

`e7_token_benchmark.py` compares five memories (transcript, summary, retrieval, state,
state+llm) on identical evidence and records the token counts the API reports for every
call, plus decision accuracy against a synthetic ground truth.

To run it with Claude Code: install it (see
https://docs.claude.com/en/docs/claude-code/overview), open a terminal in this folder,
set `ANTHROPIC_API_KEY`, start `claude`, and ask it to follow `CLAUDE.md`. Claude Code
will run the tests, an offline check, an 18-call pilot, and then ask before each larger
run. Results go to `RESULTS_e7.md`.

Offline estimates (mock backend, 4 items per session, batched extraction): the state
memory uses about 0.6x the transcript's tokens at 10 sessions and 0.13x at 60, but
simple retrieval is cheaper still, so the real test is decision quality at a given
token budget. Without batching (`--per-item-extract`) the state memory costs MORE than
the transcript for short histories, because every call repeats the instructions.

## Using your own data

Order-effect data: build `qds.models.OrderData(ab=[AyBy, AyBn, AnBy, AnBn],
ba=[ByAy, ByAn, BnAy, BnAn], name=...)` from raw counts and call
`models.q_test(data)` and `models.compare(data)`.

Review-schedule studies: replace the synthetic participants in `e1_zeno_schedule.py`
with observed responses, keep the two-stage fit (fit on the no-judgment group,
predict the rest), and compare against the Bayesian baseline.

## Correction log

- Agent memory role axes: an earlier version used the x, y, z axes, under which
  engineering and operations evidence initially had no effect on P(approve). All role
  axes now have a y-component (test: `test_every_role_moves_the_decision`).
- Paper Table 2, rank-one column: an earlier script listed the reverse-order cells
  in the wrong order. `e0_reanalysis.py` and the corrected paper use the right
  ordering (deviances 26–235).

- Experiments 7-9: an earlier comment said the state 'matches the ground truth'. It does not: with exact extraction its decision
  equals the additive truth only 69% of the time at 10 sessions and 48% at 30 (state-scale 1.0), because the rotations do not commute
  (e10, part 3). The gap seen in the API runs is therefore mostly the state's functional form, not LLM extraction noise.
