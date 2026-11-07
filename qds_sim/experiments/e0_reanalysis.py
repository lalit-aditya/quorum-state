"""Experiment 0 - Reanalysis of six published order-effect data sets (paper Table 2).

Data: Wang & Busemeyer (2013), Topics in Cognitive Science 5:689-710, Table 1
(four Gallup split-ballot surveys reviewed by Moore 2002, plus two lab studies).
For each data set: q-test, deviance G2 and AIC for the order-free, QQ-constrained
and saturated models, a rank-one quantum model in 3-D, and the reciprocity check
P(B|A) vs P(A|B) that rank-one models force to be equal (paper Lemma 1).
These are REAL published data.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from _common import out_dir, parse_args, save_csv, save_json
from qds import models


def unit(t, p):
    return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])


def rank1_3d_loglik(data, rng, restarts):
    def cells(x):
        psi = unit(x[0], x[1]); a = np.array([0, 0, 1.0]); b = unit(x[2], x[3])
        PA, PB, I = np.outer(a, a), np.outer(b, b), np.eye(3)
        s = lambda P1, P2: np.linalg.norm(P2 @ P1 @ psi) ** 2
        return (np.array([s(PA, PB), s(PA, I - PB), s(I - PA, PB), s(I - PA, I - PB)]),
                np.array([s(PB, PA), s(PB, I - PA), s(I - PB, PA), s(I - PB, I - PA)]))
    def nll(x):
        ab, ba = cells(x)
        return -(np.sum(data.ab * np.log(np.clip(ab, 1e-12, 1))) + np.sum(data.ba * np.log(np.clip(ba, 1e-12, 1))))
    best = min((minimize(nll, rng.uniform(0, np.pi, 4), method="Nelder-Mead",
                         options=dict(maxiter=4000, xatol=1e-8, fatol=1e-10)) for _ in range(restarts)),
               key=lambda r: r.fun)
    return -best.fun


def main():
    args = parse_args(__doc__.splitlines()[0])
    rng = np.random.default_rng(args.seed)
    d = out_dir(args, "e0_reanalysis")
    rows = []
    for data in models.PUBLISHED:
        q, z = models.q_test(data)
        res = models.compare(data)
        ll_sat = models.fit_saturated(data)
        ll_r1 = rank1_3d_loglik(data, rng, 20 if args.quick else 60)
        g2_r1 = 2 * (ll_sat - ll_r1)
        ab, ba = data.ab / data.ab.sum(), data.ba / data.ba.sum()
        rows.append(dict(
            data_set=data.name, N_AB=int(round(data.ab.sum())), N_BA=int(round(data.ba.sum())),
            q=round(q, 4), z=round(z, 2),
            G2_order_free=round(res["order_free"]["G2"], 2), p_order_free=round(res["order_free"]["p"], 4),
            G2_qq=round(res["qq"]["G2"], 2), p_qq=round(res["qq"]["p"], 4),
            G2_rank1=round(g2_r1, 1), p_rank1=float(chi2.sf(g2_r1, 2)),
            P_B_given_A=round(ab[0] / (ab[0] + ab[1]), 3), P_A_given_B=round(ba[0] / (ba[0] + ba[1]), 3),
            best_AIC=res["best_aic"]))
    save_csv(f"{d}/table2_reanalysis.csv", rows)
    save_json(f"{d}/summary.json", rows)
    hdr = f"{'data set':26s} {'q':>7s} {'z':>6s} {'G2 free':>8s} {'G2 QQ':>7s} {'G2 r1':>7s} {'P(B|A)':>7s} {'P(A|B)':>7s}  best AIC"
    print(hdr)
    for r in rows:
        print(f"{r['data_set']:26s} {r['q']:+7.3f} {r['z']:+6.2f} {r['G2_order_free']:8.1f} {r['G2_qq']:7.2f} "
              f"{r['G2_rank1']:7.1f} {r['P_B_given_A']:7.2f} {r['P_A_given_B']:7.2f}  {r['best_AIC']}")
    print(f"outputs -> {d}")


if __name__ == "__main__":
    main()
