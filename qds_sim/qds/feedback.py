"""Closed-loop coordination of coupled roles (paper Section 5).

Linear-Gaussian surrogate after Yamamoto, Nurdin, James & Petersen (2008):

    dx = A x dt + B dw,         dV/dt = A V + V A^T + D
    y  = C x + v,               E[dv^2] = (r / eta) dt
    u  = g y   applied along f  ->  A_cl = A + g f C,  D_cl = D + g^2 (r/eta) f f^T

Each role r is a pair x_r = (q_r, p_r): commitment displacement and its rate.
The analytic covariance is cross-checked by an Euler-Maruyama simulation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov

SIGMA = np.array([[0.0, 1.0], [-1.0, 0.0]])


@dataclass
class TwoRoleSystem:
    gamma: float = 0.5      # damping of each role toward its default
    omega: float = 1.0      # natural oscillation of commitment
    r: float = 0.25         # observation noise intensity
    eta: float = 1.0        # observation efficiency in (0, 1]

    def matrices(self):
        A1 = -self.gamma / 2 * np.eye(2) + self.omega * SIGMA
        A = np.block([[A1, np.zeros((2, 2))], [np.zeros((2, 2)), A1]])
        D = self.gamma * np.eye(4)
        C = np.array([[-1.0, 0.0, 1.0, 0.0]])   # observe disagreement q2 - q1
        f = np.array([[1.0], [0.0], [-1.0], [0.0]])  # push q1 up, q2 down
        return A, D, C, f

    def closed_loop(self, g: float):
        A, D, C, f = self.matrices()
        Acl = A + g * f @ C
        Dcl = D + g**2 * (self.r / self.eta) * f @ f.T
        return Acl, Dcl

    def is_stable(self, g: float) -> bool:
        Acl, _ = self.closed_loop(g)
        return bool(np.max(np.linalg.eigvals(Acl).real) < 0)

    def steady_covariance(self, g: float) -> np.ndarray:
        Acl, Dcl = self.closed_loop(g)
        return solve_continuous_lyapunov(Acl, -Dcl)

    def covariance_at(self, g: float, V0: np.ndarray, t: float) -> np.ndarray:
        """Exact transient covariance V(t) from V(0)=V0."""
        Acl, Dcl = self.closed_loop(g)
        Vinf = solve_continuous_lyapunov(Acl, -Dcl)
        E = expm(Acl * t)
        return E @ (V0 - Vinf) @ E.T + Vinf


def disagreement(V: np.ndarray) -> float:
    """Var(q1 - q2)."""
    return float(V[0, 0] + V[2, 2] - 2 * V[0, 2])


def mutual_information(V: np.ndarray) -> float:
    """Gaussian mutual information between role blocks (nats), paper Eq. 9."""
    return float(0.5 * np.log(np.linalg.det(V[:2, :2]) * np.linalg.det(V[2:, 2:]) / np.linalg.det(V)))


def coordination_death_time(sys: TwoRoleSystem, g: float, V0: np.ndarray,
                            threshold: float, t_max: float = 20.0, n: int = 2001):
    """First time mutual information drops below `threshold` (None if never)."""
    for t in np.linspace(0, t_max, n):
        if mutual_information(sys.covariance_at(g, V0, t)) < threshold:
            return float(t)
    return None


def simulate_paths(sys: TwoRoleSystem, g: float, n_paths: int = 4000, t_end: float = 30.0,
                   dt: float = 0.01, rng: np.random.Generator | None = None) -> np.ndarray:
    """Euler-Maruyama simulation of the closed loop; returns final states.

    The measured signal includes observation noise that is fed back, so the
    simulated covariance should match `steady_covariance` (cross-check).
    """
    rng = rng or np.random.default_rng(0)
    A, D, C, f = sys.matrices()
    x = np.zeros((n_paths, 4))
    sq_D = np.sqrt(sys.gamma)
    sq_r = np.sqrt(sys.r / sys.eta)
    for _ in range(int(t_end / dt)):
        y_dt = (x @ C.T) * dt + sq_r * rng.normal(size=(n_paths, 1)) * np.sqrt(dt)
        x = x + (x @ A.T) * dt + g * y_dt @ f.T + sq_D * rng.normal(size=x.shape) * np.sqrt(dt)
    return x
