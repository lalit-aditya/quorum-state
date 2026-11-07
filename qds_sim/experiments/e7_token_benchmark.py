"""Experiment 7 - Token benchmark: does the decision-state memory reduce tokens?

An agent reviews projects over several sessions. Three roles (finance, engineering,
operations) contribute evidence sentences each session. After every session the
agent must state P(approve). Five memory schemes are compared on the SAME evidence:

  transcript   every decision prompt contains all evidence so far
  summary      LLM keeps a <=60-word running summary (one extra call per session);
               the decision prompt contains summary + this session's evidence
  retrieval    decision prompt contains the 2 most recent items per role
  state        LLM reads each session's evidence once (JSON: role, direction,
               strength; batched per session by default, --per-item-extract for one
               call per item);
               the quantum decision state is updated locally and P(approve) is
               read from the state - no LLM call at decision time
  state+llm    same state, but the LLM makes the decision from a 1-line state
               description (tests whether the state can feed an LLM cheaply)

Measured per scheme: input/output tokens as reported by the API (`usage`), number
of calls, and decision quality against a synthetic ground truth (approve iff the
equal-weight sum of signed evidence strengths is positive), plus agreement with
the transcript scheme.

Backends:
  --backend mock       offline; token counts are ESTIMATES (~4 chars/token) and the
                       'LLM' is a heuristic. Use only to test the pipeline.
  --backend anthropic  real Claude calls. Needs `pip install anthropic` and the
                       environment variable ANTHROPIC_API_KEY. Prints the planned
                       number of calls first; --yes skips the confirmation prompt.

Docs: https://docs.claude.com/en/api/overview
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np

from _common import out_dir, parse_args, plt, save_csv, save_json
from e6_agent_memory import QuantumMemory, make_evidence

SCHEMES = ["transcript", "summary", "retrieval", "state", "state+llm"]

MIN_MAX_TOKENS = 1024     # output cap per call (see AnthropicBackend.ask)
SYSTEM = "You are a careful project-approval analyst. Follow output formats exactly."
EXTRACT = ("Read one sentence of project evidence. Reply with JSON only, no prose: "
           '{"role": "finance"|"engineering"|"operations", "direction": 1 or -1 '
           '(1 if it supports approving the project), "strength": number between 0 and 1}.\n'
           "Sentence: ")
DECIDE = ("Should this project be approved? Weigh all the evidence you are given.\n"
          "Reply with JSON only: {\"p_approve\": number between 0 and 1}.\n\n")
EXTRACT_BATCH = ("Read these sentences of project evidence. Reply with a JSON list only, one object per "
                 'sentence in order: [{"role": "finance"|"engineering"|"operations", "direction": 1 or -1 '
                 '(1 if it supports approving the project), "strength": number between 0 and 1}, ...].\n'
                 "Sentences:\n")
SUMMARISE = ("Update the running summary of project evidence so it stays under 60 words and "
             "keeps the overall balance of positive and negative evidence per role.\n"
             "Reply with the new summary text only.\n\n")


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class Usage:
    def __init__(self):
        self.calls = 0
        self.inp = 0
        self.out = 0

    def add(self, i, o):
        self.calls += 1; self.inp += i; self.out += o

    def as_dict(self):
        return dict(calls=self.calls, input_tokens=self.inp, output_tokens=self.out,
                    total_tokens=self.inp + self.out)


class AnthropicBackend:
    def __init__(self, model, max_calls, client=None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic(max_retries=6)
        self.client, self.model, self.max_calls, self.n = client, model, max_calls, 0
        self.estimated = False
        self.use_temperature = True
        self.use_no_thinking = True
        self.truncated = 0          # replies that hit max_tokens
        self.empty = 0              # replies with no text at all

    def ask(self, prompt, max_tokens=120):
        if self.n >= self.max_calls:
            raise RuntimeError(f"--max-calls {self.max_calls} reached; stopping to protect your budget")
        self.n += 1
        # Models that 'think' before answering spend output tokens on thinking, so a
        # tight cap can cut the JSON off. Give every call a generous budget; we are
        # billed only for tokens actually produced, and all of them are counted.
        kwargs = dict(model=self.model, max_tokens=max(max_tokens, MIN_MAX_TOKENS), system=SYSTEM,
                      messages=[{"role": "user", "content": prompt}])
        if self.use_temperature:
            kwargs["temperature"] = 0
        if self.use_no_thinking:
            kwargs["thinking"] = {"type": "disabled"}
        for _ in range(3):
            try:
                msg = self.client.messages.create(**kwargs)
                break
            except Exception as e:
                # Some SDK/model versions reject these options; drop the one named and retry.
                s = str(e)
                if "temperature" in s and "temperature" in kwargs:
                    self.use_temperature = False; kwargs.pop("temperature")
                    print("note: temperature not accepted; continuing without it")
                elif "thinking" in s and "thinking" in kwargs:
                    self.use_no_thinking = False; kwargs.pop("thinking")
                    print("note: cannot disable thinking for this model; thinking tokens will be counted")
                else:
                    raise
        else:
            raise RuntimeError("could not build an accepted request")
        text = "".join(getattr(b, "text", "") or "" for b in msg.content
                       if getattr(b, "type", "text") == "text")
        if getattr(msg, "stop_reason", None) == "max_tokens":
            self.truncated += 1
        if not text.strip():
            self.empty += 1
        return text, msg.usage.input_tokens, msg.usage.output_tokens


class MockBackend:
    """Offline stand-in. Token counts are estimates; answers are heuristic."""
    estimated = True
    truncated = 0
    empty = 0

    def __init__(self, rng):
        self.rng = rng

    @staticmethod
    def _tok(s):
        return max(1, len(s) // 4)

    def ask(self, prompt, max_tokens=120):
        if prompt.startswith(EXTRACT_BATCH[:20]):
            sents = [l[2:] for l in prompt.split("Sentences:\n", 1)[1].splitlines() if l.startswith("- ")]
            objs = [json.loads(self.ask(EXTRACT + s)[0]) for s in sents]
            out = json.dumps(objs)
        elif prompt.startswith(EXTRACT[:20]):
            s = prompt.split("Sentence: ", 1)[1]
            role = "finance" if s.startswith("Finance") else "engineering" if s.startswith("Engineering") else "operations"
            pos = any(w in s for w in ("under", "fell", "forward", "strong"))
            m = re.search(r"(\d+)", s)
            st = min(1.0, (int(m.group(1)) / 30) if m else 0.35)
            out = json.dumps({"role": role, "direction": 1 if pos else -1, "strength": round(st, 2)})
        elif prompt.startswith(SUMMARISE[:20]):
            out = "Summary: " + " ".join(prompt.split()[-40:])
        else:
            pos = len(re.findall(r"under|fell|forward|strong|net positive|leaning approve", prompt))
            neg = len(re.findall(r"over|rose|back|weak|net negative|leaning reject", prompt))
            p = 1 / (1 + np.exp(-(pos - neg) * 0.4))
            out = json.dumps({"p_approve": round(float(p), 3)})
        return out, self._tok(SYSTEM + prompt), self._tok(out)


def parse_json_list(text):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError(f"no JSON list in reply: {text[:120]!r}")
    return json.loads(m.group(0))


def parse_json(text, key=None):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in reply: {text[:120]!r}")
    data = json.loads(m.group(0))
    return data if key is None else data[key]


def parse_p(text, log=None):
    try:
        return float(np.clip(float(parse_json(text, "p_approve")), 0, 1))
    except Exception:
        if log is not None:
            log.append(f"decision parse error: {text[:120]!r}")
        m = re.search(r"0?\.\d+|[01](?:\.0+)?", text)
        return float(m.group(0)) if m else 0.5


# --------------------------------------------------------------------------- #
# One project
# --------------------------------------------------------------------------- #
def _extracted(d, it, log):
    try:
        role = d["role"] if d["role"] in ("finance", "engineering", "operations") else it["role"]
        sign = 1 if float(d["direction"]) >= 0 else -1
        return role, sign, float(np.clip(float(d["strength"]), 0, 1))
    except Exception as e:                      # keep going, but record it
        log.append(f"extract parse error: {e}")
        return it["role"], 1, 0.0


def run_project(backend, items, n_sessions, usage, log, batch=True):
    truth = float(np.sum([it["sign"] * it["strength"] for it in items]) > 0)
    decisions = {s: [] for s in SCHEMES}
    qmem = QuantumMemory(half_life=1e9)            # no drift. NOTE: the non-commuting rotations do NOT reproduce the order-independent additive truth (see e10)
    summary_text = "(no evidence yet)"
    extract_cost = (0, 0)
    for s in range(n_sessions):
        sess = [it for it in items if it["session"] == s]
        seen = [it for it in items if it["session"] <= s]
        sess_txt = "\n".join("- " + it["text"] for it in sess)

        # transcript
        t, i, o = backend.ask(DECIDE + "All evidence so far:\n" + "\n".join("- " + it["text"] for it in seen))
        usage["transcript"].add(i, o); decisions["transcript"].append(parse_p(t, log))

        # summary (one update call + one decision call)
        t, i, o = backend.ask(SUMMARISE + f"Current summary: {summary_text}\nNew evidence:\n{sess_txt}", 160)
        usage["summary"].add(i, o); summary_text = t.strip()
        t, i, o = backend.ask(DECIDE + f"Summary of earlier evidence: {summary_text}\nThis session:\n{sess_txt}")
        usage["summary"].add(i, o); decisions["summary"].append(parse_p(t, log))

        # retrieval: 2 most recent items per role
        picked = []
        for role in ("finance", "engineering", "operations"):
            picked += [it for it in seen if it["role"] == role][-2:]
        t, i, o = backend.ask(DECIDE + "Most recent evidence per role:\n" + "\n".join("- " + it["text"] for it in picked))
        usage["retrieval"].add(i, o); decisions["retrieval"].append(parse_p(t, log))

        # state: extraction shared by both state schemes (one call per session if batched)
        ei = eo = 0
        if batch:
            t, ei, eo = backend.ask(EXTRACT_BATCH + sess_txt, 60 * len(sess) + 40)
            try:
                objs = parse_json_list(t)
                if len(objs) != len(sess):
                    raise ValueError(f"expected {len(sess)} objects, got {len(objs)}")
            except Exception as e:
                log.append(f"batch parse error: {e}")
                objs = [{} for _ in sess]
            parsed = [_extracted(dd, it, log) for dd, it in zip(objs, sess)]
            n_calls = 1
        else:
            parsed = []
            for it in sess:
                t, i, o = backend.ask(EXTRACT + it["text"], 60)
                ei += i; eo += o
                try:
                    parsed.append(_extracted(parse_json(t), it, log))
                except Exception as e:
                    log.append(f"extract parse error: {e}"); parsed.append((it["role"], 1, 0.0))
            n_calls = len(sess)
        for it, (role, sign, strength) in zip(sess, parsed):
            qmem.add(it["text"], role, sign, strength, s)
        for name in ("state", "state+llm"):
            usage[name].calls += n_calls; usage[name].inp += ei; usage[name].out += eo
        decisions["state"].append(qmem.p_approve())
        desc = (f"Decision state after {s + 1} sessions: P(approve)={qmem.p_approve():.2f}, "
                f"P(risk acceptable)={qmem.p_risk_ok():.2f}; "
                f"{'leaning approve' if qmem.p_approve() >= 0.5 else 'leaning reject'}.")
        t, i, o = backend.ask(DECIDE + desc)
        usage["state+llm"].add(i, o); decisions["state+llm"].append(parse_p(t, log))
    return truth, {k: v[-1] for k, v in decisions.items()}, decisions


def plan_calls(P, S, E, batch=True):
    return P * S * (5 + (1 if batch else E))


def main():
    def extra(p):
        p.add_argument("--backend", choices=["mock", "anthropic"], default="mock")
        p.add_argument("--model", default=os.environ.get("QDS_MODEL", "claude-haiku-5-5"))
        p.add_argument("--projects", type=int, default=None)
        p.add_argument("--sessions", type=int, default=10)
        p.add_argument("--per-session", type=int, default=4)
        p.add_argument("--max-calls", type=int, default=3000)
        p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
        p.add_argument("--per-item-extract", action="store_true",
                       help="one extraction call per evidence item instead of one per session")
    args = parse_args(__doc__.splitlines()[0], extra)
    rng = np.random.default_rng(args.seed)
    P = args.projects or (3 if args.quick else 10)
    S, E = args.sessions, args.per_session
    batch = not args.per_item_extract
    n_calls = plan_calls(P, S, E, batch)
    d = out_dir(args, "e7_token_benchmark")

    if args.backend == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Set ANTHROPIC_API_KEY first.")
        print(f"Planned: {n_calls} API calls to {args.model} ({P} projects x {S} sessions x {E} items/session).")
        if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
            sys.exit("Cancelled.")
        backend = AnthropicBackend(args.model, args.max_calls)
    else:
        backend = MockBackend(rng)
        print(f"Mock backend: token counts are ESTIMATES. ({n_calls} simulated calls)")

    usage = {s: Usage() for s in SCHEMES}
    rows, log = [], []
    t0 = time.time()
    for proj in range(P):
        items = make_evidence(rng, S, E)
        before = {s: usage[s].inp + usage[s].out for s in SCHEMES}
        truth, final, per_session = run_project(backend, items, S, usage, log, batch)
        for s in SCHEMES:
            rows.append(dict(project=proj, scheme=s, truth=truth, p_approve=final[s],
                             correct=int((final[s] >= 0.5) == bool(truth)),
                             agrees_with_transcript=int((final[s] >= 0.5) == (final["transcript"] >= 0.5)),
                             brier=(final[s] - truth) ** 2,
                             tokens_this_project=usage[s].inp + usage[s].out - before[s]))
        print(f"project {proj + 1}/{P} done ({time.time() - t0:.0f}s)")

    summary = {}
    for s in SCHEMES:
        rr = [r for r in rows if r["scheme"] == s]
        summary[s] = dict(**usage[s].as_dict(),
                          tokens_per_project=usage[s].as_dict()["total_tokens"] / P,
                          accuracy=float(np.mean([r["correct"] for r in rr])),
                          brier=float(np.mean([r["brier"] for r in rr])),
                          agreement_with_transcript=float(np.mean([r["agrees_with_transcript"] for r in rr])))
    base = summary["transcript"]["total_tokens"]
    for s in SCHEMES:
        summary[s]["tokens_vs_transcript"] = summary[s]["total_tokens"] / base if base else float("nan")
    meta = dict(backend=args.backend, model=args.model if args.backend == "anthropic" else "mock",
                token_counts="estimated" if backend.estimated else "reported by API",
                projects=P, sessions=S, items_per_session=E, batched_extraction=batch, parse_errors=len(log),
                truncated_replies=backend.truncated, empty_replies=backend.empty,
                valid=(len(log) == 0 and backend.truncated == 0 and backend.empty == 0),
                temperature_used=getattr(backend, "use_temperature", None),
                thinking_disabled=getattr(backend, "use_no_thinking", None))
    save_csv(f"{d}/per_project.csv", rows)
    save_json(f"{d}/summary.json", dict(meta=meta, schemes=summary, parse_errors=log[:50]))

    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.6))
    axs[0].bar(SCHEMES, [summary[s]["tokens_per_project"] for s in SCHEMES], color="gray")
    axs[0].set_ylabel("tokens per project"); axs[0].tick_params(axis="x", rotation=25, labelsize=7)
    axs[1].bar(SCHEMES, [summary[s]["accuracy"] for s in SCHEMES], color="C0")
    axs[1].set_ylabel("accuracy vs ground truth"); axs[1].set_ylim(0, 1)
    axs[1].tick_params(axis="x", rotation=25, labelsize=7)
    fig.suptitle(f"{meta['model']} - tokens {meta['token_counts']}", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{d}/token_benchmark.png")

    print(f"\n[{meta['model']}; tokens {meta['token_counts']}; {P} projects x {S} sessions]")
    print(f"{'scheme':11s} {'calls':>6s} {'tokens':>9s} {'vs transcript':>13s} {'accuracy':>9s} {'Brier':>6s} {'agree':>6s}")
    for s in SCHEMES:
        v = summary[s]
        print(f"{s:11s} {v['calls']:6d} {v['total_tokens']:9d} {v['tokens_vs_transcript']:13.2f} "
              f"{v['accuracy']:9.2f} {v['brier']:6.3f} {v['agreement_with_transcript']:6.2f}")
    if not meta["valid"]:
        print(f"\n*** RUN NOT VALID: {len(log)} parse errors, {backend.truncated} truncated and "
              f"{backend.empty} empty replies (see summary.json). Do not use these numbers. ***")
    else:
        print("\nRun valid: 0 parse errors, 0 truncated, 0 empty replies.")
    print(f"outputs -> {d}")


if __name__ == "__main__":
    main()
