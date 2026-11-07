"""Experiment 5 - Model recovery and power for split-ballot order studies.

Before collecting real data, check that the analysis can tell the models apart.
Synthetic data sets are generated from three processes:
  quantum    random 4-D state with rank-2 'yes' subspaces (obeys QQ, has order effects)
  classical  one joint distribution for both orders (no order effects)
  anchor-sym   classical carry-over, same pull after 'yes' and 'no'
  anchor-asym  classical carry-over, different pulls

Key finding to report: symmetric classical carry-over satisfies the QQ equality
EXACTLY (q = 0 algebraically), so passing the q-test does not by itself favour
a quantum model (consistent with Kang 2026). The test does catch asymmetric
carry-over. Evidence for the framework therefore has to come from the joint
pattern of predictions H1-H5, not from QQ alone.
Each data set is fitted with the order-free, QQ-constrained and saturated
models; we record which model wins on AIC (confusion matrix) and how often the
q-test rejects QQ, as a function of the number of participants per order.
"""
import numpy as np

from _common import out_dir, parse_args, plt, save_csv, save_json
from qds import models

GENS = {"quantum": models.gen_quantum,
        "classical": models.gen_classical,
        "anchor-sym": models.gen_anchoring,
        "anchor-asym": lambda rng, a, b: models.gen_anchoring(rng, a, b, asymmetric=True)}
FITS = ["order_free", "qq", "saturated"]


def main():
    args = parse_args(__doc__.splitlines()[0])
    rng = np.random.default_rng(args.seed)
    d = out_dir(args, "e5_model_recovery")
    reps = 80 if args.quick else 300
    Ns = [200, 500, 1000]
    rows, conf = [], {}
    for N in Ns:
        for gname, gen in GENS.items():
            wins = dict.fromkeys(FITS, 0)
            rej = 0
            for _ in range(reps):
                data = gen(rng, N, N)
                res = models.compare(data)
                wins[res["best_aic"]] += 1
                _, z = models.q_test(data)
                rej += abs(z) > 1.96
            conf[(N, gname)] = {k: v / reps for k, v in wins.items()}
            rows.append(dict(N_per_order=N, generator=gname, reps=reps, qq_rejection_rate=rej / reps,
                             **{f"aic_win_{k}": v / reps for k, v in wins.items()}))
    save_csv(f"{d}/recovery.csv", rows)
    save_json(f"{d}/summary.json", rows)

    fig, axs = plt.subplots(1, len(Ns), figsize=(7.4, 2.7))
    for ax, N in zip(axs, Ns):
        M = np.array([[conf[(N, g)][f] for f in FITS] for g in GENS])
        ng = len(GENS)
        ax.imshow(M, vmin=0, vmax=1, cmap="Blues")
        for i in range(ng):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > 0.6 else "black")
        ax.set_xticks(range(3)); ax.set_xticklabels(["order-free", "QQ", "saturated"], fontsize=6.5)
        ax.set_yticks(range(ng)); ax.set_yticklabels(list(GENS), fontsize=6.5)
        ax.set_title(f"N = {N} per order", fontsize=8)
    axs[0].set_ylabel("true generator"); axs[1].set_xlabel("model with lowest AIC")
    fig.tight_layout(); fig.savefig(f"{d}/confusion.png")
    for r in rows:
        print(f"N={r['N_per_order']:4d} {r['generator']:11s} wins: order-free {r['aic_win_order_free']:.2f} "
              f"QQ {r['aic_win_qq']:.2f} saturated {r['aic_win_saturated']:.2f} | QQ rejected {r['qq_rejection_rate']:.2f}")
    print(f"outputs -> {d}")


if __name__ == "__main__":
    main()
