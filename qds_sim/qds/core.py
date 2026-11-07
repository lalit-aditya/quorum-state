"""Core quantum-probability operations for the decision-state framework.

Everything here is ordinary linear algebra on classical hardware. States are
density matrices (Hermitian, positive, trace 1). The four kinds of operation in
the paper are implemented as functions that map a density matrix to a new one:

    evidence   -> unitary rotation        rho -> U rho U^dagger
    commitment -> Lueders projection      rho -> P rho P / Tr(P rho)
    review     -> unsharp measurement     Kraus operators M_+, M_-
    drift      -> amplitude damping / dephasing toward a default state
"""
from __future__ import annotations

import numpy as np

# Pauli matrices
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = (SX, SY, SZ)


# --------------------------------------------------------------------------- #
# States
# --------------------------------------------------------------------------- #
def ket(*amps) -> np.ndarray:
    """Normalised column vector from amplitudes."""
    v = np.asarray(amps, dtype=complex)
    return v / np.linalg.norm(v)


def dm(v: np.ndarray) -> np.ndarray:
    """Density matrix |v><v| of a (normalised) vector."""
    v = np.asarray(v, dtype=complex).reshape(-1)
    v = v / np.linalg.norm(v)
    return np.outer(v, v.conj())


def bloch_state(theta: float, phi: float = 0.0) -> np.ndarray:
    """Pure qubit state cos(theta/2)|0> + e^{i phi} sin(theta/2)|1> as rho."""
    return dm([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])


def random_pure_state(d: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=d) + 1j * rng.normal(size=d)
    return dm(v)


def is_valid_state(rho: np.ndarray, tol: float = 1e-9) -> bool:
    herm = np.allclose(rho, rho.conj().T, atol=tol)
    tr1 = abs(np.trace(rho) - 1) < tol
    psd = np.min(np.linalg.eigvalsh((rho + rho.conj().T) / 2)) > -tol
    return bool(herm and tr1 and psd)


# --------------------------------------------------------------------------- #
# Operators
# --------------------------------------------------------------------------- #
def projector(v: np.ndarray) -> np.ndarray:
    """Rank-one projector onto vector v."""
    return dm(v)


def subspace_projector(vectors) -> np.ndarray:
    """Projector onto the span of the given vectors (orthonormalised via QR)."""
    M = np.column_stack([np.asarray(v, dtype=complex) for v in vectors])
    Q, _ = np.linalg.qr(M)
    return Q @ Q.conj().T


def axis_generator(n) -> np.ndarray:
    """Qubit generator n . sigma for a unit (or non-unit) 3-vector n."""
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    return n[0] * SX + n[1] * SY + n[2] * SZ


def unitary(G: np.ndarray, theta: float) -> np.ndarray:
    """U = exp(-i theta G / 2) for Hermitian G (eigendecomposition, any dim)."""
    w, V = np.linalg.eigh(G)
    return V @ np.diag(np.exp(-1j * theta * w / 2)) @ V.conj().T


def rotation(n, theta: float) -> np.ndarray:
    """Qubit rotation about axis n by angle theta."""
    return unitary(axis_generator(n), theta)


def commutator_norm(A: np.ndarray, B: np.ndarray) -> float:
    """Operator (spectral) norm of [A, B]; the paper's role-conflict metric."""
    return float(np.linalg.norm(A @ B - B @ A, 2))


# --------------------------------------------------------------------------- #
# Operations on states
# --------------------------------------------------------------------------- #
def evolve(rho: np.ndarray, U: np.ndarray) -> np.ndarray:
    return U @ rho @ U.conj().T


def prob(rho: np.ndarray, P: np.ndarray) -> float:
    return float(np.real(np.trace(P @ rho)))


def lueders(rho: np.ndarray, P: np.ndarray) -> tuple[float, np.ndarray]:
    """Probability of outcome P and the post-measurement (Lueders) state."""
    p = prob(rho, P)
    if p < 1e-15:
        return 0.0, rho
    return p, P @ rho @ P / p


def review_kraus(P: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray]:
    """Unsharp review of intensity s in [0, 1] (paper Eq. 4).

    s = 1 -> projective commitment, s = 0 -> no disturbance.
    Returns (M_plus, M_minus) with M+^dag M+ + M-^dag M- = I.
    """
    I = np.eye(P.shape[0])
    a, b = np.sqrt((1 + s) / 2), np.sqrt((1 - s) / 2)
    return a * P + b * (I - P), b * P + a * (I - P)


def apply_review(rho, P, s, rng=None):
    """Selective review: sample an outcome (needs rng) and update; or, if rng is
    None, apply the non-selective (averaged) channel."""
    Mp, Mm = review_kraus(P, s)
    if rng is None:
        return Mp @ rho @ Mp.conj().T + Mm @ rho @ Mm.conj().T
    pp = float(np.real(np.trace(Mp @ rho @ Mp.conj().T)))
    M = Mp if rng.random() < pp else Mm
    out = M @ rho @ M.conj().T
    return out / np.real(np.trace(out))


def amplitude_damping(rho: np.ndarray, gamma: float, default: int = 0) -> np.ndarray:
    """Qubit amplitude damping toward basis state |default> with strength gamma."""
    gamma = float(np.clip(gamma, 0.0, 1.0))
    if default == 0:
        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    else:
        K0 = np.array([[np.sqrt(1 - gamma), 0], [0, 1]], dtype=complex)
        K1 = np.array([[0, 0], [np.sqrt(gamma), 0]], dtype=complex)
    return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T


def dephasing(rho: np.ndarray, p: float) -> np.ndarray:
    """Qubit dephasing in the computational basis with probability p."""
    p = float(np.clip(p, 0.0, 1.0))
    return (1 - p) * rho + p * (SZ @ rho @ SZ)


# --------------------------------------------------------------------------- #
# Sequential question probabilities and the QQ statistic
# --------------------------------------------------------------------------- #
def sequence_probs(rho, PA, PB, between=None) -> dict:
    """Joint probabilities for answering A then B (keys 'yy','yn','ny','nn').

    `between` is an optional function rho -> rho applied after the first
    answer (evidence, reminder, or drift inserted between the questions).
    """
    I = np.eye(rho.shape[0])
    out = {}
    for a, P1 in (("y", PA), ("n", I - PA)):
        p1, r = lueders(rho, P1)
        if p1 > 0 and between is not None:
            r = between(r)
        for b, P2 in (("y", PB), ("n", I - PB)):
            out[a + b] = p1 * prob(r, P2) if p1 > 0 else 0.0
    return out


def qq_statistic(rho, PA, PB, between=None) -> float:
    """q = p_AB(different answers) - p_BA(different answers). Zero under QQ."""
    ab = sequence_probs(rho, PA, PB, between)
    ba = sequence_probs(rho, PB, PA, between)
    return (ab["yn"] + ab["ny"]) - (ba["yn"] + ba["ny"])
