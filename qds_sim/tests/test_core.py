"""Checks that the code reproduces the propositions and published numbers.

Run:  python tests/test_core.py     (or: python -m pytest -q)
"""
import numpy as np

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qds import core, drift, feedback, models


def close(a, b, tol, rel=False):
    return abs(a - b) <= (tol * abs(b) if rel else tol)


def test_same_axis_rotations_commute():          # Proposition 1
    rho = core.bloch_state(1.4)
    U1, U2 = core.rotation([0, 1, 0], -0.3), core.rotation([0, 1, 0], -0.4)
    a = core.evolve(core.evolve(rho, U1), U2)
    b = core.evolve(core.evolve(rho, U2), U1)
    assert np.allclose(a, b)


def test_distinct_axes_give_order_effects():
    rho = core.bloch_state(1.4)
    U1, U2 = core.rotation([0, 1, 0], -0.3), core.rotation([1, 0, 0], -0.4)
    P = core.projector([1, 0])
    a = core.prob(core.evolve(core.evolve(rho, U1), U2), P)
    b = core.prob(core.evolve(core.evolve(rho, U2), U1), P)
    assert abs(a - b) > 1e-3


def test_commutator_norm_is_2sinphi():            # Proposition 2
    Ga = core.axis_generator([0, 1, 0])
    for phi in np.linspace(0, np.pi, 9):
        Gb = core.axis_generator([np.sin(phi), np.cos(phi), 0])
        assert close(core.commutator_norm(Ga, Gb), 2 * abs(np.sin(phi)), 1e-9)


def test_rank_one_reciprocity():                  # Lemma 1
    rng = np.random.default_rng(3)
    for _ in range(20):
        d = rng.integers(2, 6)
        rho = core.random_pure_state(d, rng)
        a = rng.normal(size=d) + 1j * rng.normal(size=d)
        b = rng.normal(size=d) + 1j * rng.normal(size=d)
        PA, PB = core.projector(a), core.projector(b)
        _, ra = core.lueders(rho, PA)
        _, rb = core.lueders(rho, PB)
        assert close(core.prob(ra, PB), core.prob(rb, PA), 1e-9)


def test_qq_equality_holds_without_insertions():
    rng = np.random.default_rng(4)
    for _ in range(30):
        rho, PA, PB = models.random_quantum_model(rng, d=rng.integers(2, 6), k=1)
        assert close(core.qq_statistic(rho, PA, PB), 0, 1e-10)
        rho, PA, PB = models.random_quantum_model(rng, d=5, k=2)
        assert close(core.qq_statistic(rho, PA, PB), 0, 1e-10)


def test_insertions_break_qq():
    rho = core.bloch_state(1.4)
    PA, PB = core.projector([1, 0]), core.projector([1, 1])
    U = core.rotation([0, 1, 0], 0.6)
    q_ev = core.qq_statistic(rho, PA, PB, lambda r: core.evolve(r, U))
    q_dr = core.qq_statistic(rho, PA, PB, lambda r: core.amplitude_damping(r, 0.3))
    assert close(q_ev, -0.5646, 1e-3)
    assert close(q_dr, 0.1478, 1e-3)


def test_review_kraus_complete_and_limits():
    P = core.projector([1, 0])
    for s in (0.0, 0.3, 1.0):
        Mp, Mm = core.review_kraus(P, s)
        assert np.allclose(Mp.conj().T @ Mp + Mm.conj().T @ Mm, np.eye(2))
    rho = core.bloch_state(1.0)
    assert np.allclose(core.apply_review(rho, P, 0.0), rho)


def test_white_noise_null():                      # Proposition 3
    env = drift.Environment.flat()
    for tau in (0.2, 1.0, 10.0):
        assert close(drift.relative_rate(env, 0.0, tau), 1.0, 5e-3)


def test_zeno_and_antizeno_regimes():
    env = drift.Environment.lorentzian(xi=1.0, w0=0.0)
    assert drift.relative_rate(env, 0.0, 0.1) < 0.1          # resonant: Zeno
    assert drift.relative_rate(env, 3.0, 0.84) > 2.0         # detuned: anti-Zeno


def test_lyapunov_matches_monte_carlo():
    sys = feedback.TwoRoleSystem()
    V = sys.steady_covariance(0.6)
    x = feedback.simulate_paths(sys, 0.6, n_paths=6000, t_end=25, dt=0.005)
    assert close(feedback.disagreement(np.cov(x.T)), feedback.disagreement(V), 0.08, rel=True)


def test_symmetric_anchoring_satisfies_qq_exactly():
    rng = np.random.default_rng(5)
    for _ in range(20):
        a, b, k = rng.uniform(0.1, 0.9, 3)
        cells = lambda p1, p2: np.array([p1 * (p2 + k * (1 - p2)), p1 * (1 - p2 - k * (1 - p2)),
                                         (1 - p1) * p2 * (1 - k), (1 - p1) * (1 - p2 * (1 - k))])
        ab, ba = cells(a, b), cells(b, a)
        assert close((ab[1] + ab[2]) - (ba[1] + ba[2]), 0.0, 1e-12)


def test_every_role_moves_the_decision():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    from e6_agent_memory import QuantumMemory, ROLES
    for role in ROLES:
        m = QuantumMemory()
        m.add("", role, 1, 0.4, 0)
        assert m.p_approve() > 0.55, role


def test_published_reanalysis():
    rj = models.PUBLISHED[3]
    q, z = models.q_test(rj)
    assert close(z, 5.30, 0.02)
    res = models.compare(rj)
    assert close(res["qq"]["G2"], 28.56, 0.05)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in tests:
        try:
            f(); print(f"PASS  {n}")
        except AssertionError as e:
            failed += 1; print(f"FAIL  {n}  {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
