"""Models for split-ballot order-effect data (paper Section 7.1).

Data for two yes/no questions asked in both orders are eight counts:
    AB order: [AyBy, AyBn, AnBy, AnBn]
    BA order: [ByAy, ByAn, BnAy, BnAn]

Fitted models (maximum likelihood):
    order_free   3 params  one joint distribution shared by both orders (classical, no order effect)
    qq           5 params  free except the QQ equality  p_AB(diff) = p_BA(diff)
    saturated    6 params  each order free

Generators (for model recovery and power analysis):
    gen_quantum   d-dim state, rank-k 'yes' subspaces, Lueders updates (obeys QQ)
    gen_classical order-free joint distribution
    gen_anchoring classical Markov-style carry-over: the first answer shifts the second
                  (symmetric version satisfies QQ exactly; asymmetric does not)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from .core import lueders, prob, random_pure_state, subspace_projector

# index map from BA-order cells to AB-order cells of the same joint event
# BA: [ByAy, ByAn, BnAy, BnAn] corresponds to AB joint [AyBy, AnBy, AyBn, AnBn]
_BA_TO_AB = [0, 2, 1, 3]


@dataclass
class OrderData:
    ab: np.ndarray  # counts AB order
    ba: np.ndarray  # counts BA order
    name: str = ""

    @staticmethod
    def from_props(name, n_ab, p_ab, n_ba, p_ba):
        p_ab = np.asarray(p_ab, float); p_ba = np.asarray(p_ba, float)
        return OrderData(p_ab / p_ab.sum() * n_ab, p_ba / p_ba.sum() * n_ba, name)


def _ll(counts, p):
    return float(np.sum(counts * np.log(np.clip(p, 1e-12, 1))))


def q_test(d: OrderData) -> tuple[float, float]:
    """QQ statistic q and its z-score (Wang & Busemeyer 2013)."""
    nab, nba = d.ab.sum(), d.ba.sum()
    pab = (d.ab[1] + d.ab[2]) / nab
    pba = (d.ba[1] + d.ba[2]) / nba
    q = pab - pba
    p = (pab * nab + pba * nba) / (nab + nba)
    se = np.sqrt(p * (1 - p) * (1 / nab + 1 / nba))
    return float(q), float(q / se) if se > 0 else 0.0


def fit_saturated(d: OrderData) -> float:
    return _ll(d.ab, d.ab / d.ab.sum()) + _ll(d.ba, d.ba / d.ba.sum())


def fit_order_free(d: OrderData) -> float:
    joint = (d.ab + d.ba[_BA_TO_AB]) / (d.ab.sum() + d.ba.sum())
    return _ll(d.ab, joint) + _ll(d.ba, joint[_BA_TO_AB])


def fit_qq(d: OrderData) -> float:
    """Max log-likelihood under the QQ equality (closed form via a 1-D search).

    Within each order the 'same' and 'different' cells are split freely, so the
    constrained MLE only needs the common probability of 'different'.
    """
    nab, nba = d.ab.sum(), d.ba.sum()
    dab, dba = d.ab[1] + d.ab[2], d.ba[1] + d.ba[2]

    def negll(pd):
        pd = float(np.clip(pd, 1e-9, 1 - 1e-9))
        ll = 0.0
        for c, n, diff in ((d.ab, nab, dab), (d.ba, nba, dba)):
            same = n - diff
            cells = np.array([c[0] / max(same, 1e-12) * (1 - pd), c[1] / max(diff, 1e-12) * pd,
                              c[2] / max(diff, 1e-12) * pd, c[3] / max(same, 1e-12) * (1 - pd)])
            ll += _ll(c, cells)
        return -ll

    grid = np.linspace(1e-4, 1 - 1e-4, 2001)
    x0 = grid[np.argmin([negll(g) for g in grid])]
    res = minimize(lambda v: negll(v[0]), [x0], method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-12))
    return -float(res.fun)


MODELS = {"order_free": (fit_order_free, 3), "qq": (fit_qq, 5), "saturated": (fit_saturated, 6)}


def compare(d: OrderData) -> dict:
    """Log-likelihood, AIC, deviance G2 and its p-value for each model."""
    lls = fit_saturated(d)
    out = {}
    for name, (fn, k) in MODELS.items():
        ll = fn(d)
        g2 = 2 * (lls - ll)
        df = 6 - k
        out[name] = dict(ll=ll, aic=2 * k - 2 * ll, G2=g2, df=df,
                         p=float(chi2.sf(g2, df)) if df > 0 else 1.0)
    out["best_aic"] = min(MODELS, key=lambda m: out[m]["aic"])
    return out


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #
def quantum_cells(rho, PA, PB):
    I = np.eye(rho.shape[0])
    def seq(P1, P2):
        p1, r = lueders(rho, P1)
        return p1 * prob(r, P2) if p1 > 0 else 0.0
    ab = [seq(PA, PB), seq(PA, I - PB), seq(I - PA, PB), seq(I - PA, I - PB)]
    ba = [seq(PB, PA), seq(PB, I - PA), seq(I - PB, PA), seq(I - PB, I - PA)]
    return np.array(ab), np.array(ba)


def random_quantum_model(rng, d=4, k=2):
    """Random state and two rank-k 'yes' subspaces in d dimensions."""
    rho = random_pure_state(d, rng)
    def rand_sub():
        vs = [rng.normal(size=d) + 1j * rng.normal(size=d) for _ in range(k)]
        return subspace_projector(vs)
    return rho, rand_sub(), rand_sub()


def gen_quantum(rng, n_ab, n_ba, d=4, k=2) -> OrderData:
    rho, PA, PB = random_quantum_model(rng, d, k)
    pab, pba = quantum_cells(rho, PA, PB)
    return OrderData(rng.multinomial(n_ab, pab / pab.sum()).astype(float),
                     rng.multinomial(n_ba, pba / pba.sum()).astype(float), "quantum")


def gen_classical(rng, n_ab, n_ba) -> OrderData:
    joint = rng.dirichlet(np.ones(4) * 2)
    return OrderData(rng.multinomial(n_ab, joint).astype(float),
                     rng.multinomial(n_ba, joint[_BA_TO_AB]).astype(float), "classical")


def gen_anchoring(rng, n_ab, n_ba, k=None, asymmetric=False) -> OrderData:
    """Classical carry-over: after answering the first question, the second
    'yes' probability moves toward the first answer.

    Symmetric carry-over (same pull k after 'yes' and after 'no') satisfies the
    QQ equality exactly, so the q-test cannot reject it. With asymmetric pulls
    (k_yes != k_no) the equality is generally violated.
    """
    a, b = rng.uniform(0.2, 0.8, size=2)
    k = rng.uniform(0.1, 0.5) if k is None else k
    k_yes, k_no = (k, k) if not asymmetric else tuple(rng.uniform(0.0, 0.6, size=2))
    def cells(p1, p2):
        y2_after_y = p2 + k_yes * (1 - p2)
        y2_after_n = p2 * (1 - k_no)
        return np.array([p1 * y2_after_y, p1 * (1 - y2_after_y), (1 - p1) * y2_after_n, (1 - p1) * (1 - y2_after_n)])
    return OrderData(rng.multinomial(n_ab, cells(a, b)).astype(float),
                     rng.multinomial(n_ba, cells(b, a)).astype(float), "anchoring")


# The six data sets reanalysed in the paper (Wang & Busemeyer 2013, Table 1)
PUBLISHED = [
    OrderData.from_props("Clinton-Gore", 447, [.4899, .0447, .1767, .2886], 432, [.5625, .1991, .0255, .2130]),
    OrderData.from_props("Gingrich-Dole", 437, [.3936, .0824, .3204, .2037], 377, [.3448, .3422, .0637, .2493]),
    OrderData.from_props("White-Black", 459, [.3987, .0174, .1612, .4227], 486, [.4012, .0597, .1379, .4012]),
    OrderData.from_props("Rose-Jackson", 506, [.3379, .3241, .0178, .3202], 462, [.4156, .0671, .1234, .3939]),
    OrderData.from_props("Racial hostility (lab)", 116, [.2500, .0603, .2931, .3966], 108, [.3241, .0648, .2130, .3981]),
    OrderData.from_props("Affirmative action (lab)", 118, [.4831, .0424, .1610, .3136], 106, [.5660, .1321, .0283, .2736]),
]
