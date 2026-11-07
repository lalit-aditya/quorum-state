"""Redraw accuracy.png from an existing e11 summary.json (no API calls).
Usage: python replot_e11.py results/e11_accuracy_eval"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1] if len(sys.argv) > 1 else os.path.join("results", "e11_accuracy_eval")
J = json.load(open(os.path.join(d, "summary.json")))
acc, P = J["accuracy"], J["meta"]["projects"]
CODE = {"rule-code", "additive-code", "state-exact-code"}
cells = [(m, c) for m in acc for c in acc[m]]
fig, axes = plt.subplots(1, len(cells), figsize=(4.2 * len(cells), 3.6), squeeze=False)
for ax, (m, c) in zip(axes[0], cells):
    tab = {k: v for k, v in acc[m][c].items() if not k.startswith("_")}
    sch = list(tab)
    xs = np.arange(len(sch))
    a = [tab[s]["accuracy"] for s in sch]
    err = [[max(0.0, tab[s]["accuracy"] - tab[s]["accuracy_ci95"][0]) for s in sch],
           [max(0.0, tab[s]["accuracy_ci95"][1] - tab[s]["accuracy"]) for s in sch]]
    ax.bar(xs, a, yerr=err, capsize=2, color=["#555" if s in CODE else "#2a7f7f" for s in sch])
    ax.axhline(0.5, ls=":", c="k", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(sch, rotation=60, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05); ax.set_title(f"truth: {m}, {c} sessions (n={P})"); ax.set_ylabel("accuracy")
fig.tight_layout()
fig.savefig(os.path.join(d, "accuracy.png"), dpi=150)
print("wrote", os.path.join(d, "accuracy.png"))
