"""Experiment 1 - Review schedules: decision Zeno (H1) and anti-Zeno (H2).

Part A (H1). Simulated participants read 12 pieces of weak evidence that a
suspect is guilty (after Yearsley & Pothos 2016). Each piece rotates the state
toward 'guilty'. Groups make 0, 1, 2, 3, 5 or 11 intermediate judgments.
  * Quantum model: each judgment is a review (projective by default) that
    collapses the state, so frequent judgments slow belief change (Zeno).
  * Bayesian baseline: judgments do not change beliefs.
We also run the paper's two-stage analysis: fit the evidence strength on the
no-judgment group only, then predict the other groups with no further fitting.

Part B (H2). Participants hold a committed decision for time T while an
information environment pushes it toward the default. Reviews happen every
tau. Drift rates come from the overlap law (qds.drift). Two environments with
matched unreviewed drift: autocorrelated (Lorentzian) vs white noise (flat).
Two decision types: resonant with the environment, or detuned from it.
  Prediction: in the white-noise environment the schedule has no effect; in the
  autocorrelated one, detuned decisions show an anti-Zeno peak near tau ~ tau_c.

All data here are SYNTHETIC, generated from the models, to check the analysis
pipeline and estimate effect sizes before running the real study.
"""
import numpy as np

from _common import binom_ci, out_dir, parse_args, plt, save_csv, save_json
from qds import core, drift

N_PIECES = 12
GROUPS = [0, 1, 2, 3, 5, 11]          # number of intermediate judgments


def judgment_points(m):
    if m == 0:
        return set()
    step = N_PIECES // (m + 1)
    return {step * j for j in range(1, m + 1)}


def run_quantum_participant(delta, m, s, eps, rng):
    P = core.projector([1, 0])            # 'guilty'
    rho = core.dm([0, 1])                 # starts at 'innocent'
    pts = judgment_points(m)
    U = core.rotation([0, 1, 0], delta)
    for k in range(1, N_PIECES + 1):
        rho = core.evolve(rho, U)
        if k in pts and k < N_PIECES:
            rho = core.apply_review(rho, P, s, rng)
    p = core.prob(rho, P)
    resp = rng.random() < p
    if rng.random() < eps:
        resp = not resp
    return int(resp), p


def predict_quantum(delta, m, s):
    """Deterministic group prediction: non-selective reviews (outcome-averaged)."""
    P = core.projector([1, 0])
    rho = core.dm([0, 1])
    U = core.rotation([0, 1, 0], delta)
    pts = judgment_points(m)
    for k in range(1, N_PIECES + 1):
        rho = core.evolve(rho, U)
        if k in pts and k < N_PIECES:
            rho = core.apply_review(rho, P, s, None)
    return core.prob(rho, P)


def part_a(args, rng, d):
    n = 150 if args.quick else 600
    delta_bar, delta_sd, s, eps = 0.13, 0.03, 1.0, 0.05
    rows, summary = [], {}
    # Bayesian baseline calibrated to match the no-judgment group mean
    p_final0 = np.mean([predict_quantum(max(rng.normal(delta_bar, delta_sd), 0.01), 0, s) for _ in range(2000)])
    lam = (np.log(p_final0 / (1 - p_final0)) - np.log(0.05 / 0.95)) / N_PIECES
    for m in GROUPS:
        qk = []
        bk = []
        for _ in range(n):
            delta = max(rng.normal(delta_bar, delta_sd), 0.01)
            r, _ = run_quantum_participant(delta, m, s, eps, rng)
            qk.append(r)
            # Bayesian: log-odds accumulate; judgments have no effect
            lo = np.log(0.05 / 0.95) + N_PIECES * lam * delta / delta_bar
            pb = 1 / (1 + np.exp(-lo))
            rb = rng.random() < pb
            bk.append(int(rb if rng.random() >= eps else not rb))
        for model, k in (("quantum", sum(qk)), ("bayesian", sum(bk))):
            lo_, hi_ = binom_ci(k, n)
            rows.append(dict(part="A", model=model, intermediate_judgments=m, n=n,
                             p_guilty=k / n, ci_low=lo_, ci_high=hi_))
    # Two-stage analysis on the quantum-generated data
    obs = {r["intermediate_judgments"]: r["p_guilty"] for r in rows if r["model"] == "quantum"}
    # P(guilty) = sin^2(12*delta/2) is periodic in delta, so one group cannot
    # identify delta uniquely. Restrict to the monotone branch 12*delta <= pi
    # (evidence never 'overshoots'); with real data, report this assumption.
    grid = np.linspace(0.005, np.pi / N_PIECES, 400)
    target0 = (obs[0] - eps) / (1 - 2 * eps)      # undo response noise
    fit_delta = grid[np.argmin([(predict_quantum(g, 0, s) - target0) ** 2 for g in grid])]
    preds = {m: eps + (1 - 2 * eps) * predict_quantum(fit_delta, m, s) for m in GROUPS}
    rmse = float(np.sqrt(np.mean([(preds[m] - obs[m]) ** 2 for m in GROUPS if m > 0])))
    summary.update(fitted_delta=float(fit_delta), true_mean_delta=delta_bar,
                   two_stage_rmse=rmse, predictions=preds, observed=obs)

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for model, c in (("quantum", "k"), ("bayesian", "C0")):
        rr = [r for r in rows if r["model"] == model]
        x = [r["intermediate_judgments"] for r in rr]
        y = [r["p_guilty"] for r in rr]
        e = [[r["p_guilty"] - r["ci_low"] for r in rr], [r["ci_high"] - r["p_guilty"] for r in rr]]
        ax.errorbar(x, y, yerr=e, fmt="o-", color=c, ms=3, capsize=2, label=f"{model} (simulated)")
    ax.plot(GROUPS, [preds[m] for m in GROUPS], "k:", label="two-stage prediction")
    ax.set_xlabel("number of intermediate judgments")
    ax.set_ylabel("P(final judgment = guilty)")
    ax.legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(f"{d}/h1_zeno.png")
    return rows, summary


def part_b(args, rng, d):
    n = 200 if args.quick else 800
    T = 20.0
    taus = [np.inf, 10, 4, 2, 1, 0.84, 0.5, 0.25, 0.1]
    lor = drift.Environment.lorentzian(xi=1.0, w0=0.0, gamma=1.0)
    groups = {"detuned": (3.0, 0.4), "resonant": (0.0, 0.2)}
    rows, summary = [], {}
    for gname, (wa_mean, wa_sd) in groups.items():
        # scale environment so the unreviewed decision keeps ~50% retention at T
        scale = (np.log(2) / T) / drift.golden_rule_rate(lor, wa_mean)
        env_l = drift.Environment(lambda w, s=scale: s * lor.G(w), "lorentzian")
        level = env_l.G(np.array([wa_mean]))[0]
        env_f = drift.Environment.flat(level)
        wa_grid = np.linspace(wa_mean - 3 * wa_sd, wa_mean + 3 * wa_sd, 9)
        for env in (env_l, env_f):
            table = {}
            for tau in taus:
                if np.isinf(tau):
                    table[tau] = np.array([drift.golden_rule_rate(env, w) for w in wa_grid])
                else:
                    table[tau] = np.array([drift.drift_rate(env, w, tau) for w in wa_grid])
            for tau in taus:
                kept = 0
                for _ in range(n):
                    wa = np.clip(rng.normal(wa_mean, wa_sd), wa_grid[0], wa_grid[-1])
                    R = np.interp(wa, wa_grid, table[tau])
                    kept += rng.random() < drift.survival(R, T)
                lo_, hi_ = binom_ci(kept, n)
                rows.append(dict(part="B", decision=gname, environment=env.name, tau=tau, n=n,
                                 retention=kept / n, ci_low=lo_, ci_high=hi_))
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
    for ax, g in zip(axs, groups):
        for env, c in (("lorentzian", "C3"), ("flat", "C0")):
            rr = [r for r in rows if r["decision"] == g and r["environment"] == env and np.isfinite(r["tau"])]
            base = [r for r in rows if r["decision"] == g and r["environment"] == env and np.isinf(r["tau"])][0]
            x = [r["tau"] for r in rr]; y = [r["retention"] for r in rr]
            e = [[r["retention"] - r["ci_low"] for r in rr], [r["ci_high"] - r["retention"] for r in rr]]
            ax.errorbar(x, y, yerr=e, fmt="o-", ms=3, capsize=2, color=c, label=f"{env} env.")
            ax.axhline(base["retention"], color=c, ls=":", lw=0.8)
        ax.set_xscale("log"); ax.set_title(f"{g} decision", fontsize=9)
        ax.set_xlabel(r"review interval $\tau$ ($\tau_c$ units)")
    axs[0].set_ylabel(f"retention after T={T:g}"); axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{d}/h2_schedule.png")
    for g in groups:
        for env in ("lorentzian", "flat"):
            rr = {r["tau"]: r["retention"] for r in rows if r["decision"] == g and r["environment"] == env}
            summary[f"{g}/{env}"] = dict(no_review=rr[np.inf], min=min(rr.values()), max=max(rr.values()))
    return rows, summary


def main():
    args = parse_args(__doc__.splitlines()[0])
    rng = np.random.default_rng(args.seed)
    d = out_dir(args, "e1_zeno_schedule")
    rows_a, sum_a = part_a(args, rng, d)
    rows_b, sum_b = part_b(args, rng, d)
    save_csv(f"{d}/h1_groups.csv", rows_a)
    save_csv(f"{d}/h2_schedule.csv", rows_b)
    save_json(f"{d}/summary.json", dict(H1=sum_a, H2=sum_b))
    print("H1 two-stage: fitted delta %.3f (true mean %.2f), prediction RMSE %.3f"
          % (sum_a["fitted_delta"], sum_a["true_mean_delta"], sum_a["two_stage_rmse"]))
    for k, v in sum_a["observed"].items():
        print(f"  judgments={k:2d}  observed P(guilty)={v:.3f}  predicted={sum_a['predictions'][k]:.3f}")
    for k, v in sum_b.items():
        print(f"H2 {k:22s} no-review {v['no_review']:.3f}  range [{v['min']:.3f}, {v['max']:.3f}]")
    print(f"outputs -> {d}")


if __name__ == "__main__":
    main()
