# Running experiment 9 (macro / meso / micro hierarchy)

The ground truth in `e9_hierarchy.py` (function `ground_truth`) is a SYNTHETIC rule written by the script
author. Replace it with your real rule (what makes 'fast-track' correct, what makes 'commit' correct)
before drawing practical conclusions.

## Setup and checks (no API cost)
    pip install numpy scipy matplotlib anthropic
    export ANTHROPIC_API_KEY=...        # Windows PowerShell: $env:ANTHROPIC_API_KEY="..."
    python tests/test_core.py && python tests/test_pipeline.py && python tests/test_hierarchy.py
    cd experiments
    python e9_hierarchy.py --demo

## Pilot with the real API (up to 120 calls: 3 projects x 5 sessions, escalations are the uncertain part)
    python e9_hierarchy.py --backend anthropic --projects 3 --sessions 5 --yes \
        --price-small IN,OUT --price-ideal IN,OUT --max-calls 150 --out ../results/e9_pilot
`valid` in summary.json must be true. Prices are USD per million tokens; the defaults are placeholders.

## Main runs (up to 700 and 1,000 calls)
    python e9_hierarchy.py --backend anthropic --projects 10 --sessions 10 --yes ... --max-calls 800 --out ../results/e9_s10
    python e9_hierarchy.py --backend anthropic --projects 10 --sessions 30 --yes ... --max-calls 1200 --out ../results/e9_s30
Add `--escalate-margin 0.05` or `0.2` to see how the gate trades calls for accuracy. Fix the margin in advance;
choosing it after seeing results on the same projects is tuning.

## Reading the table
* Compare `hier-gated` with `flat-small` and `flat-ideal`, per level and for all three levels together.
* `hier-gated` is an extra design choice of this script, not defined in the paper.
* The paper's macro re-examination rule is not implemented (its 'meso probability of reaching the goal' is not defined yet).
* W_k (macro -> meso preparation) are fixed, not learned. micro -> meso feedback is off by default.
