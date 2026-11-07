"""Experiment 4 - Feedback keeps coupled roles coordinated (H5).

Two roles (e.g. finance and engineering) each drift around their own default.
The organisation observes their disagreement with noise (efficiency eta) and
feeds it back with gain g, pushing the roles toward each other.

Predictions (paper Section 5):
  * steady disagreement has an interior optimum in g (helpful drift reshaping vs
    harmful noise injection ~ g^2);
  * the optimal gain falls as observation efficiency eta falls;
  * without feedback, coordination (mutual information) crosses an operational
    threshold in finite time ('coordination sudden death'); with feedback it
    need not.
Analytic results (Lyapunov equation) are cross-checked by stochastic
simulation (Euler-Maruyama). Data are SYNTHETIC.
"""
import numpy as np

from _common import out_dir, parse_args, plt, save_csv, save_json
from qds import feedback as fb


def main():
    args = parse_args(__doc__.splitlines()[0])
    rng = np.random.default_rng(args.seed)
    d = out_dir(args, "e4_feedback")
    etas = [1.0, 0.5, 0.2]
    gains = np.linspace(0, 3, 61)
    rows, summary = [], {}
    for eta in etas:
        sys = fb.TwoRoleSystem(eta=eta)
        dis = [fb.disagreement(sys.steady_covariance(g)) if sys.is_stable(g) else np.nan for g in gains]
        for g, v in zip(gains, dis):
            rows.append(dict(eta=eta, gain=g, disagreement_analytic=v))
        i = int(np.nanargmin(dis))
        summary[f"eta={eta}"] = dict(g_opt=float(gains[i]), disagreement_opt=float(dis[i]),
                                     disagreement_no_feedback=float(dis[0]),
                                     reduction=float(1 - dis[i] / dis[0]))
    # Monte Carlo cross-check at a few gains
    mc_rows = []
    n_paths = 1500 if args.quick else 6000
    for eta in etas:
        sys = fb.TwoRoleSystem(eta=eta)
        for g in (0.0, 0.5, 1.0, 2.0):
            x = fb.simulate_paths(sys, g, n_paths=n_paths, t_end=25, dt=0.01, rng=rng)
            mc_rows.append(dict(eta=eta, gain=g, disagreement_mc=fb.disagreement(np.cov(x.T)),
                                disagreement_analytic=fb.disagreement(sys.steady_covariance(g))))
    # Coordination sudden death vs gain
    V0 = np.array([[1, 0, .95, 0], [0, 1, 0, .95], [.95, 0, 1, 0], [0, .95, 0, 1]])
    thr = 0.2
    csd = []
    sys = fb.TwoRoleSystem()
    for g in (0.0, 0.1, 0.2, 0.4, 0.6, 1.0, 1.5):
        t = fb.coordination_death_time(sys, g, V0, thr)
        csd.append(dict(gain=g, death_time=round(t, 2) if t is not None else "never",
                        steady_MI=fb.mutual_information(sys.steady_covariance(g))))
    save_csv(f"{d}/h5_gain_sweep.csv", rows)
    save_csv(f"{d}/h5_monte_carlo.csv", mc_rows)
    save_csv(f"{d}/h5_coordination_death.csv", csd)
    save_json(f"{d}/summary.json", dict(optimum=summary, monte_carlo=mc_rows, coordination_death=csd))

    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.6))
    for eta in etas:
        rr = [r for r in rows if r["eta"] == eta]
        axs[0].plot([r["gain"] for r in rr], [r["disagreement_analytic"] for r in rr], label=fr"$\eta$={eta}")
        mm = [r for r in mc_rows if r["eta"] == eta]
        axs[0].plot([r["gain"] for r in mm], [r["disagreement_mc"] for r in mm], "x", color="k", ms=4)
    axs[0].set_xlabel("feedback gain g"); axs[0].set_ylabel(r"Var$(q_1-q_2)$  (x = Monte Carlo)")
    axs[0].legend(fontsize=7)
    ts = np.linspace(0, 12, 300)
    for g in (0.0, 0.2, 0.6):
        axs[1].plot(ts, [fb.mutual_information(sys.covariance_at(g, V0, t)) for t in ts], label=f"g={g}")
    axs[1].axhline(thr, color="gray", ls=":", lw=0.8)
    axs[1].set_xlabel("time (model units)"); axs[1].set_ylabel("mutual information (nats)")
    axs[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{d}/h5_feedback.png")

    for k, v in summary.items():
        print(f"{k}: optimal g={v['g_opt']:.2f}, disagreement {v['disagreement_no_feedback']:.2f} -> "
              f"{v['disagreement_opt']:.2f} ({100 * v['reduction']:.0f}% lower)")
    worst = max(abs(r["disagreement_mc"] - r["disagreement_analytic"]) / r["disagreement_analytic"] for r in mc_rows)
    print(f"Monte Carlo vs analytic: max relative error {100 * worst:.1f}%")
    for r in csd:
        print(f"gain {r['gain']:.1f}: coordination death at t = {r['death_time']}, steady MI {r['steady_MI']:.3f}")
    print(f"outputs -> {d}")


if __name__ == "__main__":
    main()
