"""Checks for experiments/e9_hierarchy.py (run: python tests/test_hierarchy.py)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
import numpy as np
from e6_agent_memory import make_evidence
import e8_orchestrated as e8
import e9_hierarchy as e9


class Cfg:
    half_life = 1e9; state_scale = 1.0; prep_bias = 0.0; feedback_eps = 0.0
    decide_every = 2; escalate_margin = 0.1


def item(role, sign, strength, session=0):
    return dict(role=role, sign=sign, strength=strength, session=session, text="x")


def test_ground_truth_rules():
    t = e9.ground_truth([item("finance", 1, .5), item("engineering", 1, .3), item("operations", -1, .1)])
    assert t == dict(macro="approve", meso="fast-track", micro="commit")
    t = e9.ground_truth([item("finance", -1, .5), item("engineering", 1, .3), item("operations", 1, .5)])
    assert t["macro"] == "approve" and t["meso"] == "fast-track" and t["micro"] == "defer"
    t = e9.ground_truth([item("finance", 1, .1), item("engineering", -1, .6), item("operations", -1, .2)])
    assert t == dict(macro="reject", meso="none", micro="defer")


def test_macro_level_matches_flat_state_and_roles_route_correctly():
    rng = np.random.default_rng(3)
    items = make_evidence(rng, 5, 4)
    h, flat = e9.HierState(Cfg), e8.DecisionState(1e9)
    h.lv["meso"].rho = h.lv["meso"].rho            # prep_bias = 0 -> identity preparation
    for it in items:
        h.rotate(it["role"], it["sign"], it["strength"]); flat.rotate(it["role"], it["sign"], it["strength"])
    assert abs(h.p()["macro"] - flat.p_approve()) < 1e-9
    h2 = e9.HierState(Cfg); p0 = h2.p()
    h2.rotate("finance", 1, 0.5)
    assert h2.p()["meso"] == p0["meso"]            # finance does not move the meso level
    assert h2.p()["micro"] != p0["micro"]


def test_end_to_end_mock_is_valid_and_gating_saves_calls():
    rng = np.random.default_rng(1)
    r = e8.Router(e9.HierMock(rng), e9.HierMock(rng), {"small": (1, 5), "ideal": (3, 15)}, 800)
    usage = {s: e8.SchemeUsage() for s in e9.SCHEMES}; log = []
    esc = dict(decisions=0, escalated=0, macro=0, meso=0, micro=0)
    truth, final = e9.run_project(r, make_evidence(rng, 4, 4), 4, Cfg, usage, log, esc)
    assert not log and set(final) == set(e9.SCHEMES)
    assert usage["hier-code"].calls == 4
    assert usage["hier-ideal"].calls == 4 + 2
    assert usage["hier-gated"].calls == 4 + esc["escalated"] <= 6


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("PASS ", k); n += 1
    print(f"{n}/{n} passed")
