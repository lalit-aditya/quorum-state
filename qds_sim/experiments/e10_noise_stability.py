"""Experiment 10 - Noise stability of the decision pipeline (after Haris, Zhang & Yoshida, 2026).

Haris et al. (arXiv:2602.08287, ICLR 2026) define the noise stability of a function f under rho-correlated
noise applied to ALL input coordinates at once: Stab_rho(f) = E[f(X) f(Y)], with Y a rho-correlated copy of X.
For discrete inputs their regularizer (Definition 6.1, Eq. 7) uses the Bonami-Beckner process: each coordinate
is kept with probability (1+rho)/2 and otherwise replaced by a fresh uniform draw. (With that parametrization the
coordinate correlation is (1+rho)/2, so rho = 0 is not independence.) We instead keep each item with probability
rho, so that rho IS the coordinate correlation (rho = 0 independent, rho = 1 identical).

ADAPTATION (ours, not theirs): we apply that noise operator to the EVIDENCE ITEMS that feed our decision
state, not to the weights or tokens of a Transformer. The 'function' is the final decision of a pipeline.
Haris et al. study Transformers; nothing here tests their theorems.

Part 1  Noise stability of the decision as a function of the evidence stream.
        Each evidence item (role, direction, strength) is kept w.p. (1+rho)/2, else resampled from the
        generator. d(X) = +1 / -1 is approve / reject. Compared: the additive ground-truth rule
        (sign of the equal-weight sum) and the quantum decision state (non-commuting role rotations).
Part 2  Test-retest reliability under extraction noise. The SAME evidence is 'extracted' twice with
        independent noise (direction flipped w.p. eps, Gaussian strength error sigma). We report how often the
        two decisions differ, accuracy against the clean ground truth, and how often the state is confidently
        wrong (max(p, 1-p) > 0.9). The additive rule applied to the same noisy items is the control.

Part 3  Fidelity of the state to an order-independent additive rule when extraction is EXACT. The benchmark
        ground truth (experiments 7-9) is the sign of the equal-weight sum. The state applies non-commuting
        rotations, so it is order-dependent by design. We measure how often its decision equals the additive
        truth as history grows, for several --state-scale values, and how often shuffling the evidence changes it.

Everything here is simulation on the templated evidence of experiment 6. No API calls.
Usage: python e10_noise_stability.py [--quick] [--projects N]
"""
from __future__ import annotations

import numpy as np

from _common import binom_ci, out_dir, parse_args, plt, save_csv, save_json
from e6_agent_memory import ROLES

ROLE_NAMES = list(ROLES)
AXES = [ROLES[r] for r in ROLE_NAMES]


def sample_items(rng, n):
    return dict(role=rng.integers(0, 3, n), sign=rng.choice([-1, 1], n), strength=rng.uniform(0.1, 0.6, n))


def bb_noise(rng, X, rho):
    """Bonami-Beckner noise: keep each item w.p. (1+rho)/2, else draw a fresh item."""
    n = len(X["role"])
    keep = rng.random(n) < rho
    F = sample_items(rng, n)
    return {k: np.where(keep, X[k], F[k]) for k in X}


def extraction_noise(rng, X, eps, sigma):
    n = len(X["role"])
    flip = rng.random(n) < eps
    s = np.clip(X["strength"] + rng.normal(0, sigma, n), 0.01, 1.0)
    return dict(role=X["role"], sign=np.where(flip, -X["sign"], X["sign"]), strength=s)


def state_p(X, scale=1.0):
    """P(approve) of the qubit decision state: undecided prior, rotation -sign*strength*scale about each role axis."""
    a0 = b0 = np.cos(np.pi / 4) + 0j
    b0 = np.sin(np.pi / 4) + 0j
    for r, sg, st in zip(X["role"], X["sign"], X["strength"]):
        nx, ny, nz = AXES[r]
        t = -sg * st * scale
        c, s = np.cos(t / 2), np.sin(t / 2)
        a0, b0 = ((c - 1j * s * nz) * a0 - 1j * s * (nx - 1j * ny) * b0,
                  -1j * s * (nx + 1j * ny) * a0 + (c + 1j * s * nz) * b0)
    return float(abs(a0) ** 2)


def additive_d(X):
    return 1 if np.sum(X["sign"] * X["strength"]) > 0 else -1


def state_d(X):
    return 1 if state_p(X) >= 0.5 else -1


def part1(rng, M, n_items, rhos):
    rows = []
    for rho in rhos:
        da = np.empty((M, 2)); ds = np.empty((M, 2))
        for m in range(M):
            X = sample_items(rng, n_items); Y = bb_noise(rng, X, rho)
            da[m] = additive_d(X), additive_d(Y)
            ds[m] = state_d(X), state_d(Y)
        for name, d in (("additive", da), ("state", ds)):
            stab = float(np.mean(d[:, 0] * d[:, 1]))
            k = int(np.sum(d[:, 0] == d[:, 1]))
            lo, hi = binom_ci(k, M)
            rows.append(dict(n_items=n_items, rho=rho, function=name, stab=stab, agree=k / M,
                             agree_lo=lo, agree_hi=hi, mean_decision=float(np.mean(d[:, 0]))))
    return rows


def part2(rng, M, n_items, epss, sigma):
    rows = []
    for eps in epss:
        acc = {"additive": [], "state": []}; flip = {"additive": [], "state": []}
        dp, conf_wrong, wrong = [], 0, 0
        for m in range(M):
            X = sample_items(rng, n_items); truth = additive_d(X)
            X1, X2 = extraction_noise(rng, X, eps, sigma), extraction_noise(rng, X, eps, sigma)
            p1, p2 = state_p(X1), state_p(X2)
            d = {"additive": (additive_d(X1), additive_d(X2)), "state": (1 if p1 >= .5 else -1, 1 if p2 >= .5 else -1)}
            for k in d:
                flip[k].append(int(d[k][0] != d[k][1]))
                acc[k] += [int(d[k][0] == truth), int(d[k][1] == truth)]
            dp.append(abs(p1 - p2))
            for p, dd in ((p1, d["state"][0]), (p2, d["state"][1])):
                if dd != truth:
                    wrong += 1
                    conf_wrong += int(max(p, 1 - p) > 0.9)
        for k in ("additive", "state"):
            a = int(np.sum(acc[k])); lo, hi = binom_ci(a, 2 * M)
            rows.append(dict(n_items=n_items, eps=eps, sigma=sigma, function=k, flip_rate=float(np.mean(flip[k])),
                             accuracy=a / (2 * M), acc_lo=lo, acc_hi=hi,
                             mean_abs_dp=float(np.mean(dp)) if k == "state" else float("nan"),
                             confident_wrong_share=(conf_wrong / wrong if wrong else float("nan")) if k == "state" else float("nan")))
    return rows


def part3(rng, M, sessions, scales):
    rows = []
    for S in sessions:
        for sc in scales:
            ok = ch = 0
            for _ in range(M):
                X = sample_items(rng, 4 * S)
                d = 1 if state_p(X, sc) >= 0.5 else -1
                ok += int(d == additive_d(X))
                idx = rng.permutation(4 * S)
                Y = {k: v[idx] for k, v in X.items()}
                ch += int((1 if state_p(Y, sc) >= 0.5 else -1) != d)
            lo, hi = binom_ci(ok, M)
            rows.append(dict(sessions=S, scale=sc, accuracy_vs_additive=ok / M, acc_lo=lo, acc_hi=hi,
                             shuffle_changes_decision=ch / M))
    return rows


def main():
    def extra(p):
        p.add_argument("--projects", type=int, default=None)
        p.add_argument("--sigma", type=float, default=0.10, help="strength error (std) in part 2")
    args = parse_args(__doc__.splitlines()[0], extra)
    rng = np.random.default_rng(args.seed)
    M = args.projects or (300 if args.quick else 1500)
    d = out_dir(args, "e10_noise_stability")
    rhos = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]
    epss = [0.0, 0.05, 0.10, 0.18, 0.25]
    r1, r2 = [], []
    for n_items in (40, 120):                     # 10 and 30 sessions of 4 items
        r1 += part1(rng, M, n_items, rhos)
        r2 += part2(rng, M, n_items, epss, args.sigma)
        print(f"n_items={n_items} done")
    save_csv(f"{d}/part1_noise_stability.csv", r1)
    save_csv(f"{d}/part2_extraction_noise.csv", r2)
    r3 = part3(rng, M, [1, 2, 5, 10, 30, 60], [1.0, 0.3, 0.1, 0.03])
    save_csv(f"{d}/part3_fidelity.csv", r3)
    save_json(f"{d}/summary.json", dict(projects=M, sigma=args.sigma, part1=r1, part2=r2, part3=r3))

    fig, axs = plt.subplots(1, 4, figsize=(11.5, 2.7))
    for n_items, ls in ((40, "-"), (120, "--")):
        for fn, c in (("additive", "C0"), ("state", "C3")):
            rr = [r for r in r1 if r["n_items"] == n_items and r["function"] == fn]
            axs[0].plot([r["rho"] for r in rr], [r["stab"] for r in rr], ls, color=c, label=f"{fn}, {n_items // 4} sessions")
            rr = [r for r in r2 if r["n_items"] == n_items and r["function"] == fn]
            axs[1].plot([r["eps"] for r in rr], [r["flip_rate"] for r in rr], ls, color=c)
            axs[2].plot([r["eps"] for r in rr], [r["accuracy"] for r in rr], ls, color=c)
    axs[0].set_xlabel(r"correlation $\rho$"); axs[0].set_ylabel(r"noise stability $E[d(X)d(Y)]$"); axs[0].legend(fontsize=6)
    axs[1].set_xlabel("direction-flip rate"); axs[1].set_ylabel("decision differs between 2 extractions")
    axs[2].set_xlabel("direction-flip rate"); axs[2].set_ylabel("accuracy vs clean ground truth")
    for sc in (1.0, 0.3, 0.1, 0.03):
        rr = [r for r in r3 if r["scale"] == sc]
        axs[3].plot([r["sessions"] for r in rr], [r["accuracy_vs_additive"] for r in rr], "o-", ms=3, label=f"scale {sc}")
    axs[3].set_xscale("log"); axs[3].set_xlabel("sessions of evidence"); axs[3].set_ylabel("state accuracy vs additive truth\n(exact extraction)")
    axs[3].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(f"{d}/noise_stability.png")

    print(f"\nPart 1: noise stability E[d(X)d(Y)] ({M} projects per cell)")
    print(f"{'items':>5s} {'func':>9s} " + " ".join(f"rho={r:<4}" for r in rhos))
    for n_items in (40, 120):
        for fn in ("additive", "state"):
            rr = [r for r in r1 if r["n_items"] == n_items and r["function"] == fn]
            print(f"{n_items:5d} {fn:>9s} " + " ".join(f"{r['stab']:8.2f}" for r in rr))
    print(f"\nPart 2: test-retest under extraction noise (sigma={args.sigma})")
    print(f"{'items':>5s} {'eps':>5s} {'func':>9s} {'flip':>6s} {'accuracy':>9s} {'|dp|':>6s} {'conf.wrong':>10s}")
    for r in r2:
        print(f"{r['n_items']:5d} {r['eps']:5.2f} {r['function']:>9s} {r['flip_rate']:6.2f} {r['accuracy']:9.2f} "
              f"{r['mean_abs_dp']:6.2f} {r['confident_wrong_share']:10.2f}")
    print("\nPart 3: state vs additive truth with EXACT extraction")
    print(f"{'sessions':>8s} {'scale':>6s} {'accuracy':>9s} {'shuffle changes decision':>25s}")
    for r in r3:
        print(f"{r['sessions']:8d} {r['scale']:6.2f} {r['accuracy_vs_additive']:9.2f} {r['shuffle_changes_decision']:25.2f}")
    print(f"outputs -> {d}")


if __name__ == "__main__":
    main()
