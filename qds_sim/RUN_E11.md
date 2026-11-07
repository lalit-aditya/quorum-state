# Running experiment 11 (accuracy evaluation)

Question: is the decision state as accurate as the traditional memories, and where does it pay off?

## Offline check (free, mock LLM: tests the pipeline only, says nothing about real models)
    cd experiments
    python e11_accuracy_eval.py --projects 6 --sessions 6 --checkpoints 3,6 --with-summary

## Real run (60 projects x 2 truths x 8,400 calls; prints the plan and asks before spending)
    export ANTHROPIC_API_KEY=...        # Windows PowerShell: $env:ANTHROPIC_API_KEY="..."
    python e11_accuracy_eval.py --backend anthropic --projects 60 --sessions 30 --checkpoints 10,30 --truth both --with-summary --ideal-thinking off --price-small 1,5 --price-ideal 3,15 --max-calls 9000

Set your real prices; the defaults are placeholders. Use `--projects 100` or more for a paper claim.
Output goes to `results/e11_accuracy_eval/` (summary.json, per_project.csv, accuracy.png).
If the plot fails, redraw it from the saved results: `python replot_e11.py ../results/e11_accuracy_eval`.

## Reading the result
* A scheme is "as accurate" as the baseline only if its interval overlaps, the paired p-value is large, and n is big enough to see a difference you care about.
* `state-exact-code` vs `state-code`: the gap is extraction error. The distance of `state-exact-code` from `rule-code` is the state's functional form.
* `additive-code` under the `order` truth is what an order-blind memory can reach. The state helps only if it beats that.
* Compare `output tokens per call` across schemes, and check `thinking_disabled_in_effect` in summary.json, before crediting a cost saving to the memory design.
* Run a second pass with `--ideal-thinking default` to see whether the large model's accuracy depends on its extra output.
