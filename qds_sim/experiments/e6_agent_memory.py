"""Experiment 6 - Multi-role agent memory harness (paper Study 3).

Three roles (finance -> cost, engineering -> risk, operations -> timing) feed
evidence about a project to an agent over several sessions. A language model
reads each evidence sentence and outputs (role, direction, strength). Four
memory schemes then store what was learned:

  transcript  keeps every evidence sentence (text; cost grows with history)
  summary     keeps only the last M sentences (lossy text)
  belief      classical log-odds per decision (2 numbers + timestamp)
  quantum     density matrix per decision; each role is a rotation axis,
              stored state drifts toward the default between sessions

At the end the agent answers two questions in both orders:
  A: "Approve the project?"   B: "Is the risk acceptable?"
Reported per scheme: memory footprint, P(approve), order effect, q statistic,
and EVIDENCE-ORDER sensitivity (spread of P(approve) when the same evidence
arrives in shuffled order).

IMPORTANT: with the default --backend mock, the 'LLM' is a deterministic
parser of templated sentences with added noise. Results then only demonstrate
that the pipeline works and how the schemes differ by construction; they are
NOT evidence about real agents or people. Run with --backend anthropic to use
Claude (needs the `anthropic` package, network access, and ANTHROPIC_API_KEY).
API docs: https://docs.claude.com/en/api/overview
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from _common import out_dir, parse_args, plt, save_csv, save_json
from qds import core

# Criterion axes. The undecided prior lies on the Bloch x-axis, so a rotation moves
# P(approve) only through the axis' y-component. Every role therefore has a
# y-component, and the axes are 60 degrees apart, so the roles stay non-commuting.
_C, _S = np.cos(np.pi / 3), np.sin(np.pi / 3)
ROLES = {"finance": [0.0, 1.0, 0.0], "engineering": [_S, _C, 0.0], "operations": [0.0, _C, _S]}
TEMPLATES = {
    "finance": ["Finance update: costs came in {d} budget by {p}%.", "Finance review: cash position looks {q}."],
    "engineering": ["Engineering note: test failures {d2} by {p}%.", "Engineering review: design risk looks {q}."],
    "operations": ["Ops update: schedule slipped {d3} by {p} days.", "Ops review: delivery timing looks {q}."],
}


# --------------------------------------------------------------------------- #
# Evidence generation
# --------------------------------------------------------------------------- #
def make_evidence(rng, n_sessions, per_session):
    items = []
    for s in range(n_sessions):
        for _ in range(per_session):
            role = rng.choice(list(ROLES))
            sign = rng.choice([-1, 1])
            strength = float(rng.uniform(0.1, 0.6))
            p = int(round(strength * 30))
            word = "strong" if sign > 0 else "weak"
            t = TEMPLATES[role][rng.integers(2)].format(
                d="under" if sign > 0 else "over", d2="fell" if sign > 0 else "rose",
                d3="back" if sign < 0 else "forward", p=p, q=word)
            items.append(dict(session=s, role=role, sign=sign, strength=strength, text=t))
    return items


# --------------------------------------------------------------------------- #
# LLM backends: map an evidence sentence to (role, direction, strength)
# --------------------------------------------------------------------------- #
class MockLLM:
    """Template parser + noise. Stands in for an LLM so the harness runs offline."""
    def __init__(self, rng, noise=0.05):
        self.rng, self.noise = rng, noise

    def read(self, text):
        role = ("finance" if text.startswith("Finance") else
                "engineering" if text.startswith("Engineering") else "operations")
        positive = any(w in text for w in ("under", "fell", "forward", "strong"))
        m = re.search(r"(\d+)", text)
        strength = (int(m.group(1)) / 30) if m else 0.35
        strength = float(np.clip(strength + self.rng.normal(0, self.noise), 0.05, 1.0))
        return role, (1 if positive else -1), strength


class AnthropicLLM:
    """Calls Claude through the Anthropic Messages API (untested offline)."""
    PROMPT = ("You read one sentence of project evidence. Reply with JSON only, no prose: "
              '{"role": "finance"|"engineering"|"operations", "direction": 1 or -1 '
              "(1 if it supports approving the project), \"strength\": number in [0,1]}.\n"
              "Sentence: ")

    def __init__(self, model):
        import anthropic  # noqa: WPS433 (optional dependency)
        self.client = anthropic.Anthropic()
        self.model = model

    def read(self, text):
        msg = self.client.messages.create(model=self.model, max_tokens=100,
                                          messages=[{"role": "user", "content": self.PROMPT + text}])
        raw = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        data = json.loads(raw.strip().strip("`").replace("json", "", 1))
        return data["role"], int(np.sign(data["direction"]) or 1), float(np.clip(data["strength"], 0, 1))


def tokens(text):
    return max(1, len(text) // 4)            # rough estimate: ~4 characters per token


# --------------------------------------------------------------------------- #
# Memory schemes
# --------------------------------------------------------------------------- #
class TranscriptMemory:
    name = "transcript"
    def __init__(self, **_): self.items = []
    def add(self, text, role, sign, strength, session): self.items.append((text, sign * strength))
    def footprint(self): return sum(tokens(t) for t, _ in self.items), "tokens"
    def p_approve(self): return float(1 / (1 + np.exp(-sum(v for _, v in self.items))))
    def p_risk_ok(self): return self.p_approve()


class SummaryMemory(TranscriptMemory):
    name = "summary"
    def __init__(self, keep=6, **_): super().__init__(); self.keep = keep
    def add(self, *a, **k):
        super().add(*a, **k); self.items = self.items[-self.keep:]


class BeliefMemory:
    name = "belief"
    def __init__(self, half_life=3.0, **_):
        self.lo, self.last, self.half_life = 0.0, 0, half_life
    def add(self, text, role, sign, strength, session):
        self.lo *= 0.5 ** ((session - self.last) / self.half_life); self.last = session
        self.lo += sign * strength
    def footprint(self): return 2, "numbers"
    def p_approve(self): return float(1 / (1 + np.exp(-self.lo)))
    def p_risk_ok(self): return self.p_approve()


class QuantumMemory:
    name = "quantum"
    def __init__(self, half_life=3.0, **_):
        self.rho = core.dm([np.cos(np.pi / 4), np.sin(np.pi / 4)])   # undecided prior
        self.last, self.half_life = 0, half_life
        self.P_approve = core.projector([1, 0])
        self.P_risk = core.projector([np.cos(np.pi / 8), np.sin(np.pi / 8)])   # incompatible question
    def add(self, text, role, sign, strength, session):
        gap = session - self.last
        if gap > 0:   # drift toward the default ('do not approve', |1>)
            self.rho = core.amplitude_damping(self.rho, 1 - 0.5 ** (gap / self.half_life), default=1)
            self.last = session
        self.rho = core.evolve(self.rho, core.rotation(ROLES[role], -sign * strength))
    def footprint(self): return 4, "numbers"   # 3 real Bloch components + timestamp
    def p_approve(self): return core.prob(self.rho, self.P_approve)
    def p_risk_ok(self): return core.prob(self.rho, self.P_risk)
    def order_stats(self):
        ab = core.sequence_probs(self.rho, self.P_approve, self.P_risk)
        ba = core.sequence_probs(self.rho, self.P_risk, self.P_approve)
        return (ba["yy"] + ba["ny"]) - (ab["yy"] + ab["yn"]), core.qq_statistic(self.rho, self.P_approve, self.P_risk)


SCHEMES = [TranscriptMemory, SummaryMemory, BeliefMemory, QuantumMemory]


def run_scheme(cls, readings, items):
    mem = cls()
    for it, (role, sign, strength) in zip(items, readings):
        mem.add(it["text"], role, sign, strength, it["session"])
    order_effect, q = mem.order_stats() if hasattr(mem, "order_stats") else (0.0, 0.0)
    fp, unit = mem.footprint()
    return dict(p_approve=mem.p_approve(), order_effect=order_effect, q=q, footprint=fp, unit=unit)


def main():
    def extra(p):
        p.add_argument("--backend", choices=["mock", "anthropic"], default="mock")
        p.add_argument("--model", default=os.environ.get("QDS_MODEL", "claude-sonnet-5-5"),
                       help="Claude model string (see docs.claude.com for current models)")
        p.add_argument("--sessions", type=int, default=10)
        p.add_argument("--per-session", type=int, default=4)
        p.add_argument("--projects", type=int, default=None)
        p.add_argument("--shuffles", type=int, default=None)
    args = parse_args(__doc__.splitlines()[0], extra)
    rng = np.random.default_rng(args.seed)
    d = out_dir(args, "e6_agent_memory")
    llm = MockLLM(rng) if args.backend == "mock" else AnthropicLLM(args.model)
    n_projects = args.projects or (20 if args.quick else 100)
    n_shuffles = args.shuffles or (10 if args.quick else 30)
    if args.backend == "anthropic":
        print(f"Using Claude model {args.model}: ~{n_projects * args.sessions * args.per_session} API calls")

    rows = []
    for proj in range(n_projects):
        items = make_evidence(rng, args.sessions, args.per_session)
        readings = [llm.read(it["text"]) for it in items]      # LLM reads each item once
        acc = np.mean([r[0] == it["role"] and r[1] == it["sign"] for r, it in zip(readings, items)])
        for cls in SCHEMES:
            base = run_scheme(cls, readings, items)
            # evidence-order sensitivity: shuffle order within each session, rerun
            spreads = []
            for _ in range(n_shuffles):
                idx = []
                for s in range(args.sessions):
                    block = [i for i, it in enumerate(items) if it["session"] == s]
                    idx += list(rng.permutation(block))
                spreads.append(run_scheme(cls, [readings[i] for i in idx], [items[i] for i in idx])["p_approve"])
            rows.append(dict(project=proj, scheme=cls.name, llm_read_accuracy=acc,
                             evidence_order_sd=float(np.std(spreads)), **base))
    save_csv(f"{d}/agent_memory.csv", rows)

    summary = {}
    for cls in SCHEMES:
        rr = [r for r in rows if r["scheme"] == cls.name]
        summary[cls.name] = dict(
            footprint_mean=float(np.mean([r["footprint"] for r in rr])), unit=rr[0]["unit"],
            p_approve_mean=float(np.mean([r["p_approve"] for r in rr])),
            question_order_effect_mean_abs=float(np.mean([abs(r["order_effect"]) for r in rr])),
            q_mean_abs=float(np.mean([abs(r["q"]) for r in rr])),
            evidence_order_sd_mean=float(np.mean([r["evidence_order_sd"] for r in rr])))
    summary["_note"] = ("mock backend: illustrative only" if args.backend == "mock"
                        else f"backend {args.model}")
    summary["llm_read_accuracy"] = float(np.mean([r["llm_read_accuracy"] for r in rows]))
    save_json(f"{d}/summary.json", summary)

    names = [c.name for c in SCHEMES]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
    axs[0].bar(names, [summary[n]["footprint_mean"] for n in names], color="gray")
    axs[0].set_yscale("log"); axs[0].set_ylabel("memory (tokens or numbers)")
    axs[1].bar(names, [summary[n]["question_order_effect_mean_abs"] for n in names], color="C0")
    axs[1].set_ylabel("|question-order effect|")
    axs[2].bar(names, [summary[n]["evidence_order_sd_mean"] for n in names], color="C3")
    axs[2].set_ylabel("SD of P(approve)\nover evidence orders")
    for ax in axs:
        ax.tick_params(axis="x", labelsize=6.5, rotation=20)
    fig.tight_layout(); fig.savefig(f"{d}/agent_memory.png")

    print(f"LLM read accuracy (role and direction): {summary['llm_read_accuracy']:.3f}  [{summary['_note']}]")
    for n in names:
        s = summary[n]
        print(f"{n:10s} memory {s['footprint_mean']:7.1f} {s['unit']:7s}  P(approve) {s['p_approve_mean']:.3f}  "
              f"|order effect| {s['question_order_effect_mean_abs']:.3f}  |q| {s['q_mean_abs']:.4f}  "
              f"evidence-order SD {s['evidence_order_sd_mean']:.3f}")
    print(f"outputs -> {d}")


if __name__ == "__main__":
    main()
