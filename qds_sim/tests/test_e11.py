"""Checks for experiments/e11_accuracy_eval.py (run: python tests/test_e11.py)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
import numpy as np
import e11_accuracy_eval as e11
from e6_agent_memory import make_evidence


def _it(role, sign, strength, session):
    return dict(role=role, sign=sign, strength=strength, session=session, text="x")


def test_order_truth_depends_on_order_additive_does_not():
    a = [_it("finance", 1, 0.5, 0), _it("finance", -1, 0.45, 1)]
    b = list(reversed(a))
    assert e11.truth_additive(a)[0] == e11.truth_additive(b)[0] == 1.0
    assert e11.truth_order(a)[0] != e11.truth_order(b)[0]          # recency flips it


def test_balanced_finals():
    rng = np.random.default_rng(3)
    truths = [e11.truth_order(e11.balanced_items(rng, 6, 4, float(i % 2), e11.truth_order))[0] for i in range(20)]
    assert sum(truths) == 10


def test_exact_sign_test():
    assert e11.exact_sign_test(0, 0) == 1.0
    assert e11.exact_sign_test(5, 5) == 1.0
    assert e11.exact_sign_test(10, 0) < 0.01


def test_controls_and_valid_mock_run():
    class A: half_life = 1e9; state_scale = 1.0
    rng = np.random.default_rng(1)
    r = e11.Router(e11.MockBackend(rng), e11.MockBackend(rng), {"small": (1, 5), "ideal": (3, 15)}, 5000)
    schemes = e11.LLM_SCHEMES + ["summary-small"]
    usage = {s: e11.SchemeUsage() for s in schemes}; log = []; rows = []
    for p in range(8):
        items = e11.balanced_items(rng, 6, 4, float(p % 2), e11.truth_order)
        res = e11.run_project(r, items, 6, A, "order", usage, log, {3, 6}, True)
        for cp, x in res.items():
            assert x["p"]["rule-code"] == x["truth"]                # sanity control is exact
            for s in schemes + e11.CODE_SCHEMES:
                rows.append(dict(project=p, scheme=s, truth=x["truth"], margin=x["margin"], p=float(x["p"][s]),
                                 correct=int((float(x["p"][s]) >= 0.5) == bool(x["truth"]))))
    assert not log
    assert usage["state-code"].calls == 6 * 8                        # extraction every session only
    assert usage["transcript-ideal"].calls == 2 * 8                  # one decision per checkpoint
    summ = e11.summarise(rows, schemes + e11.CODE_SCHEMES, usage, 8)
    assert summ["rule-code"]["accuracy"] == 1.0


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); n += 1; print("ok", k)
    print(f"{n} checks passed")
