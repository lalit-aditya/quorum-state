"""Experiment 3 - Inserted delay breaks the QQ equality (H4) + power analysis.

Two incompatible questions are asked with a delay Delta between them. During
the delay the state drifts toward a default:
    rho -> (1 - p) rho + p rho_default,   p = 1 - exp(-kappa * Delta).
Optionally, intervening content (evidence) also rotates the state.

Paper's H4: q = 0 for back-to-back questions and |q| grows with accumulated
drift. This simulation tests how robust that is:
  * a robustness sweep over many random models shows that drift alone does
    produce q != 0, but its size depends strongly on where the default state
    lies (it can be near zero, as in the seeded example model);
  * intervening content produces a large q that shrinks as delay grows.
So H4 holds generically but its effect size is heterogeneous; the study needs
the content manipulation, or large samples, to have power.

We use a 4-dimensional model with rank-2 'yes' subspaces, since rank-1 models
are rejected by human data (paper, Lemma 1). Then we estimate how many
participants per order are needed to detect the predicted q at each delay
(z-test at alpha = .05), and check the false-positive rate at Delta = 0.
Data are SYNTHETIC.
"""
import numpy as np

from _common import out_dir, parse_args, plt, save_csv, save_json
from qds import core, models

HALF_LIFE_H = 6.0                       # assumed drift half-life (hours)
DELAYS = {"0 s": 0.0, "30 s": 30 / 3600, "5 min": 5 / 60, "1 h": 1.0, "24 h": 24.0}


def build_model(seed=7, d=4):
    """Pick a seeded random model with moderate marginals and an order effect."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(4000):
        rho, PA, PB = models.random_quantum_model(rng, d=d, k=2)
        pa, pb = core.prob(rho, PA), core.prob(rho, PB)
        ab = core.sequence_probs(rho, PA, PB)
        ba = core.sequence_probs(rho, PB, PA)
        effect = abs((ba["yy"] + ba["ny"]) - pa)          # order effect on A
        if 0.35 < pa < 0.65 and 0.35 < pb < 0.65 and (best is None or effect > best[0]):
            best = (effect, rho, PA, PB)
    _, rho, PA, PB = best
    v = (np.eye(d) - PA) @ (rng.normal(size=d) + 1j * rng.normal(size=d))
    rho_def = core.dm(v)
    G = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    G = (G + G.conj().T) / 2
    return rho, PA, PB, rho_def, G


def between_fn(delta_h, rho_def, G=None, content_angle=0.0):
    p = 1 - np.exp(-np.log(2) / HALF_LIFE_H * delta_h)
    U = core.unitary(G, content_angle) if G is not None and content_angle else None

    def f(r):
        if U is not None:
            r = core.evolve(r, U)
        return (1 - p) * r + p * rho_def
    return f


def cells(rho, PA, PB, fn):
    ab = core.sequence_probs(rho, PA, PB, fn)
    ba = core.sequence_probs(rho, PB, PA, fn)
    pab = np.array([ab["yy"], ab["yn"], ab["ny"], ab["nn"]])
    pba = np.array([ba["yy"], ba["yn"], ba["ny"], ba["nn"]])   # ByAy, ByAn, BnAy, BnAn
    return pab / pab.sum(), pba / pba.sum()


def main():
    args = parse_args(__doc__.splitlines()[0])
    rng = np.random.default_rng(args.seed)
    d = out_dir(args, "e3_delay_qq")
    rho, PA, PB, rho_def, G = build_model()
    sims = 200 if args.quick else 1000
    Ns = [100, 250, 500, 1000]
    rows, power_rows = [], []
    for cond, angle in (("delay only", 0.0), ("delay + content", 0.8)):
        for lab, h in DELAYS.items():
            fn = between_fn(h, rho_def, G, angle)
            pab, pba = cells(rho, PA, PB, fn)
            q = (pab[1] + pab[2]) - (pba[1] + pba[2])
            pa_first = pab[0] + pab[1]
            pa_second = pba[0] + pba[2]
            rows.append(dict(condition=cond, delay=lab, hours=h, q=q,
                             order_effect_A=pa_second - pa_first))
            for N in Ns:
                rej = 0
                for _ in range(sims):
                    dat = models.OrderData(rng.multinomial(N, pab).astype(float),
                                           rng.multinomial(N, pba).astype(float))
                    _, z = models.q_test(dat)
                    rej += abs(z) > 1.96
                power_rows.append(dict(condition=cond, delay=lab, N_per_order=N, true_q=q,
                                       rejection_rate=rej / sims))
    save_csv(f"{d}/h4_q_by_delay.csv", rows)
    save_csv(f"{d}/h4_power.csv", power_rows)

    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.6))
    labs = list(DELAYS)
    for cond, c in (("delay only", "k"), ("delay + content", "C3")):
        rr = [r for r in rows if r["condition"] == cond]
        axs[0].plot(labs, [r["q"] for r in rr], "o-", color=c, label=f"q, {cond}")
        axs[0].plot(labs, [r["order_effect_A"] for r in rr], "s--", color=c, ms=3, label=f"order effect, {cond}")
    axs[0].axhline(0, color="gray", lw=0.5); axs[0].set_xlabel("delay between questions"); axs[0].legend(fontsize=6)
    for N, ls in zip(Ns, ["-", "--", "-.", ":"]):
        rr = [r for r in power_rows if r["condition"] == "delay + content" and r["N_per_order"] == N]
        axs[1].plot(labs, [r["rejection_rate"] for r in rr], ls, marker="o", ms=3, label=f"N={N}/order")
    axs[1].axhline(0.05, color="gray", lw=0.5); axs[1].axhline(0.8, color="gray", lw=0.5, ls=":")
    axs[1].set_ylabel("P(reject q = 0)"); axs[1].set_xlabel("delay (delay + content condition)"); axs[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(f"{d}/h4_delay_qq.png")

    # robustness: distribution of |q| over random models and random defaults
    rob = []
    n_models = 150 if args.quick else 600
    for p in (0.1, 0.5, 0.94):
        qs = []
        for _ in range(n_models):
            r_, A_, B_ = models.random_quantum_model(rng, 4, 2)
            dflt = core.dm(rng.normal(size=4) + 1j * rng.normal(size=4))
            qs.append(abs(core.qq_statistic(r_, A_, B_, lambda r, p=p, dd=dflt: (1 - p) * r + p * dd)))
        rob.append(dict(drift_fraction=p, median_abs_q=float(np.median(qs)),
                        p90_abs_q=float(np.percentile(qs, 90)), n_models=n_models))
    save_csv(f"{d}/h4_robustness.csv", rob)
    for r in rob:
        print(f"robustness: drift fraction {r['drift_fraction']:.2f}  median |q| {r['median_abs_q']:.3f}  "
              f"90th pct {r['p90_abs_q']:.3f}")

    summary = dict(robustness=rob, model=dict(P_A=core.prob(rho, PA), P_B=core.prob(rho, PB), half_life_hours=HALF_LIFE_H),
                   q=rows, power=power_rows)
    save_json(f"{d}/summary.json", summary)
    for r in rows:
        print(f"{r['condition']:16s} {r['delay']:>6s}  q={r['q']:+.4f}  order effect on A={r['order_effect_A']:+.4f}")
    for cond in ("delay only", "delay + content"):
        print(f"power ({cond}):  rejection rate of q = 0 by N per order")
        for lab in DELAYS:
            rr = [r for r in power_rows if r["condition"] == cond and r["delay"] == lab]
            print(f"  {lab:>6s}  " + "  ".join(f"N={r['N_per_order']}: {r['rejection_rate']:.2f}" for r in rr))
    print(f"outputs -> {d}")


if __name__ == "__main__":
    main()
