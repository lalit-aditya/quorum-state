"""Experiment 2 - Reminders versus recommitment (H3).

A committed decision |yes> is stored for time T. The environment perturbs it in
two ways:
  * a transverse 'pressure' (rotation about a random axis in the x-y plane,
    rate ~ N(0, sigma)) that is redrawn at random times with switching rate nu
    (nu = 0: quasi-static pressure; large nu: fast-fluctuating pressure), and
  * weak amplitude damping toward the default |no>.
With a budget of K interventions, compare:
  none        - no intervention
  recommit    - K forced recommitments (projective review on the yes/no basis)
  reminder    - K reminders: coherent pi-rotations about the commitment axis
                (a 'spin echo'; it re-presents the rationale without asking for
                an answer, and is reversible)
Outcomes at time T:
  retention   P(yes) when finally asked
  order_sens  order sensitivity of a later pair of incompatible questions

Kofman & Kurizki (2005) showed for physical decay that coherent modulation can
match measurement for suppressing decay while staying reversible. H3 asks
whether the same holds for stored decisions. The physics also predicts the
boundary: coherent refocusing only works if interventions are faster than the
environment's correlation time, so reminders should lose their advantage as nu
grows. Data are SYNTHETIC.
"""
import numpy as np

from _common import out_dir, parse_args, plt, save_csv, save_json
from qds import core

P_YES = core.projector([1, 0])


def evolve_segment(rho, U, gamma_seg):
    return core.amplitude_damping(core.evolve(rho, U), gamma_seg, default=1)


def run(strategy, K, sigma, nu, kappa, T, n_steps, rng):
    dt = T / n_steps

    def draw():
        phi = rng.uniform(0, 2 * np.pi)
        return core.rotation([np.cos(phi), np.sin(phi), 0.0], rng.normal(0, sigma) * dt)
    U = draw()
    g = 1 - np.exp(-kappa * dt)
    rho = core.dm([1, 0])
    # intervention times: midpoints of K equal windows (CPMG-like spacing)
    times = set(int(round((j + 0.5) * n_steps / K)) for j in range(K)) if K > 0 else set()
    Z_pi = core.rotation([0, 0, 1], np.pi)
    for step in range(1, n_steps + 1):
        if nu > 0 and rng.random() < nu * dt:
            U = draw()
        rho = evolve_segment(rho, U, g)
        if step in times:
            if strategy == "recommit":
                rho = core.apply_review(rho, P_YES, 1.0, rng)
            elif strategy == "reminder":
                rho = core.evolve(rho, Z_pi)
    retention = core.prob(rho, P_YES)
    # later incompatible pair: A = yes/no, B = tilted question at 60 degrees
    PB = core.projector([np.cos(np.pi / 6), np.sin(np.pi / 6)])
    ab = core.sequence_probs(rho, P_YES, PB)
    ba = core.sequence_probs(rho, PB, P_YES)
    order_sens = abs(ab["yy"] - ba["yy"])
    return retention, order_sens


def main():
    args = parse_args(__doc__.splitlines()[0])
    rng = np.random.default_rng(args.seed)
    d = out_dir(args, "e2_reminders")
    n = 200 if args.quick else 1000
    T, n_steps, kappa = 10.0, 200, 0.03
    sigma = 0.5
    nus = [0.0, 0.5, 2.0, 8.0]
    budgets = [0, 1, 2, 4, 8, 16]
    rows = []
    for nu in nus:
        for K in budgets:
            for strat in (["none"] if K == 0 else ["recommit", "reminder"]):
                res = np.array([run(strat, K, sigma, nu, kappa, T, n_steps, rng) for _ in range(n)])
                m, se = res.mean(0), res.std(0) / np.sqrt(n)
                rows.append(dict(sigma=sigma, nu=nu, budget=K, strategy=strat, n=n,
                                 retention=m[0], retention_se=se[0],
                                 order_sens=m[1], order_sens_se=se[1]))
    save_csv(f"{d}/h3_results.csv", rows)
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.1))
    for ax, key, lab in zip(axs, ["retention", "order_sens"], ["retention P(yes)", "later order sensitivity"]):
        for nu, ls in zip(nus, ["-", "--", "-.", ":"]):
            for strat, c in (("recommit", "C3"), ("reminder", "C0")):
                rr = sorted([r for r in rows if r["nu"] == nu and r["strategy"] in (strat, "none")],
                            key=lambda r: r["budget"])
                ax.plot([r["budget"] for r in rr], [r[key] for r in rr], ls, color=c,
                        label=f"{strat}, $\\nu$={nu}" if key == "retention" else None)
        ax.set_xlabel("intervention budget K"); ax.set_ylabel(lab)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6, ncol=4, loc="lower center", frameon=False)
    fig.tight_layout(rect=(0, 0.17, 1, 1)); fig.savefig(f"{d}/h3_reminders.png")
    summary = {}
    for nu in nus:
        for K in (4, 16):
            get = lambda s, k=K: [r for r in rows if r["nu"] == nu and r["budget"] == (0 if s == "none" else k)
                                  and r["strategy"] == s][0]
            summary[f"nu={nu},K={K}"] = {s: {k: round(float(get(s)[k]), 3) for k in ("retention", "order_sens")}
                                         for s in ("none", "recommit", "reminder")}
    save_json(f"{d}/summary.json", summary)
    for k, v in summary.items():
        print(k, v)
    print(f"outputs -> {d}")


if __name__ == "__main__":
    main()
