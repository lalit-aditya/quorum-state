"""Schedule-dependent drift of a stored decision (paper Section 4).

Implements the weak-coupling overlap law of Kofman & Kurizki (2001, 2005):

    R(tau) = 2*pi * integral G(w + w_a) F_tau(w) dw,
    F_tau(w) = (tau / 2 pi) sinc^2(w tau / 2)       (periodic projective review)

G is the spectrum of the information environment, w_a the decision's
characteristic rate. Rates are reported relative to the unreviewed
("golden-rule") rate R_GR = 2*pi*G(w_a).

Validity: weak coupling, i.e. claims restricted to regimes where the stored
decision is still likely to survive.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

_W = np.linspace(-3000.0, 3000.0, 1_200_001)
_DW = _W[1] - _W[0]


@dataclass
class Environment:
    """Information-environment spectrum G(w)."""
    G: Callable[[np.ndarray], np.ndarray]
    name: str = "env"

    @staticmethod
    def lorentzian(xi: float = 1.0, w0: float = 0.0, gamma: float = 1.0) -> "Environment":
        """Autocorrelated environment with correlation time 1/xi peaked at w0."""
        def G(w):
            return gamma / (2 * np.pi) * xi**2 / ((w - w0) ** 2 + xi**2)
        return Environment(G, f"lorentzian(xi={xi}, w0={w0})")

    @staticmethod
    def flat(level: float = 1.0 / (2 * np.pi)) -> "Environment":
        """White-noise environment: the paper's null case (Proposition 3)."""
        return Environment(lambda w: np.full_like(np.asarray(w, float), level), "flat")


def sinc_filter(tau: float, w: np.ndarray = _W) -> np.ndarray:
    """Normalised spectrum F_tau of periodic projective review at interval tau."""
    x = w * tau / 2
    return tau * np.sinc(x / np.pi) ** 2 / (2 * np.pi)


def golden_rule_rate(env: Environment, w_a: float) -> float:
    return float(2 * np.pi * env.G(np.array([w_a]))[0])


def drift_rate(env: Environment, w_a: float, tau: float) -> float:
    """Absolute drift rate R(tau) under periodic review at interval tau."""
    return float(2 * np.pi * np.sum(env.G(_W + w_a) * sinc_filter(tau)) * _DW)


def relative_rate(env: Environment, w_a: float, tau: float) -> float:
    return drift_rate(env, w_a, tau) / golden_rule_rate(env, w_a)


def schedule_rate(env: Environment, w_a: float, intervals: Sequence[float]) -> float:
    """Effective rate for an arbitrary schedule of projective reviews.

    Each review restarts the evolution, so the decay over a window of length
    tau_k proceeds at R(tau_k). The time-averaged rate is
        R_eff = sum_k tau_k R(tau_k) / sum_k tau_k.
    """
    intervals = np.asarray(intervals, float)
    rates = np.array([drift_rate(env, w_a, t) for t in intervals])
    return float(np.sum(intervals * rates) / np.sum(intervals))


def survival(rate: float, t: float) -> float:
    """Probability that a committed decision is still held after time t."""
    return float(np.exp(-rate * t))
