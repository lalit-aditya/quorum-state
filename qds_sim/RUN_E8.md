# Running experiment 8 (step-by-step framework with a small and an ideal model)

## 0. Setup
    pip install numpy scipy matplotlib anthropic
    export ANTHROPIC_API_KEY=...           # never paste the key into a file or a chat
    python tests/test_core.py && python tests/test_pipeline.py     # 13/13 and 3/3 must pass

## 1. See every step, no API calls
    cd experiments
    python e8_orchestrated.py --demo

## 2. Pilot with the real API (75 calls)
    python e8_orchestrated.py --backend anthropic --projects 3 --sessions 5 --yes \
        --price-small IN,OUT --price-ideal IN,OUT --out ../results/e8_pilot
Check summary.json: `valid` must be true (0 parse errors, 0 truncated, 0 empty replies) and
token counts must say "reported by API". Models default to claude-haiku-5-5 (small) and
claude-sonnet-5-5 (ideal); change with --small-model / --ideal-model or QDS_SMALL_MODEL / QDS_IDEAL_MODEL.
Replace IN,OUT with your real USD per million tokens (input,output); the defaults are placeholders.

## 3. Main runs (ask yourself before each; use --max-calls as a hard cap)
    # frequent decisions, medium history: 10 projects x 10 sessions = 500 calls
    python e8_orchestrated.py --backend anthropic --projects 10 --sessions 10 --yes ... --out ../results/e8_s10
    # long history: 5 projects x 30 sessions = 750 calls
    python e8_orchestrated.py --backend anthropic --projects 5 --sessions 30 --yes ... --out ../results/e8_s30
    # rare decisions (the case where the transcript is cheapest): answer every 5th session
    python e8_orchestrated.py --backend anthropic --projects 5 --sessions 30 --decide-every 5 --yes ... --out ../results/e8_s30_d5
For conclusions use 50-100 projects per setting and fresh --seed values.

## 4. How to read the table
* Compare `orchestrated` with BOTH `transcript-ideal` and `transcript-small`. The honest question is whether
  the state pipeline beats the cheapest traditional memory that is equally accurate, not only the expensive one.
* `retrieval-small` is usually the cheapest; check its accuracy.
* Do not change --state-scale after seeing results on the same projects. Report any tuned run as exploratory.
* Results are on templated evidence with an additive ground truth, which favours additive memories.
