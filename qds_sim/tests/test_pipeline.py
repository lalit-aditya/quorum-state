"""Checks for experiments/e8_orchestrated.py (run: python tests/test_pipeline.py)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
import numpy as np
from e6_agent_memory import QuantumMemory, make_evidence
import e8_orchestrated as e8


def test_state_matches_benchmark_state():
    """The step-by-step state must give the same P(approve) as the e7 QuantumMemory."""
    rng = np.random.default_rng(7)
    items = make_evidence(rng, 6, 4)
    ref, new = QuantumMemory(half_life=1e9), e8.DecisionState(half_life=1e9)
    for s in range(6):
        if s > 0:
            new.drift(1.0)
        for it in [i for i in items if i["session"] == s]:
            ref.add(it["text"], it["role"], it["sign"], it["strength"], s)
            new.rotate(it["role"], it["sign"], it["strength"])
    assert abs(ref.p_approve() - new.p_approve()) < 1e-6


def test_review_s0_is_identity_and_drift_moves_toward_default():
    st = e8.DecisionState(half_life=2.0)
    st.rotate("finance", 1, 0.5)
    p0 = st.p_approve(); st.review(0.0)
    assert abs(st.p_approve() - p0) < 1e-12
    st.drift(10.0)
    assert st.p_approve() < p0                      # default is 'do not approve'


def test_end_to_end_mock_run_is_valid():
    class A: decide_every = 2; half_life = 1e9; state_scale = 1.0; review_every = 0
    A.review_intensity = 0.5; A.feedback_gain = 0.0
    rng = np.random.default_rng(1)
    r = e8.Router(e8.MockBackend(rng), e8.MockBackend(rng), {"small": (1, 5), "ideal": (3, 15)}, 500)
    usage = {s: e8.SchemeUsage() for s in e8.SCHEMES}; log = []
    truth, final, trace = e8.run_project(r, make_evidence(rng, 4, 4), 4, A, usage, log)
    assert not log and set(final) == set(e8.SCHEMES)
    assert usage["state-code"].calls == 4            # extraction only, no answer calls
    assert usage["orchestrated"].calls == 4 + 2      # extraction + 2 answers (sessions 2 and 4)
    assert usage["transcript-ideal"].calls == 2


def test_noise_operator_and_fast_state_agree_with_library():
    import e10_noise_stability as e10
    rng = np.random.default_rng(2)
    X = e10.sample_items(rng, 30)
    st = e8.DecisionState(1e9)
    for r, sg, s_ in zip(X["role"], X["sign"], X["strength"]):
        st.rotate(e10.ROLE_NAMES[r], int(sg), float(s_))
    assert abs(st.p_approve() - e10.state_p(X)) < 1e-9
    same = e10.bb_noise(rng, X, 1.0)
    assert all(np.array_equal(same[k], X[k]) for k in X)             # rho = 1: identical copy


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("PASS ", k); n += 1
    print(f"{n}/{n} passed")
