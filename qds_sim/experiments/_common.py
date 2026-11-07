"""Shared helpers: output folders, plotting style, CLI flags, saving results."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 9, "font.family": "serif", "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})


def parse_args(description: str, extra=None):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--quick", action="store_true", help="smaller run for a fast check")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default=None, help="output folder (default results/<script>)")
    if extra:
        extra(p)
    return p.parse_args()


def out_dir(args, name: str) -> str:
    d = args.out or os.path.join(ROOT, "results", name)
    os.makedirs(d, exist_ok=True)
    return d


def save_json(path, obj):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=conv)


def save_csv(path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def binom_ci(k, n, z=1.96):
    """Wilson 95% interval for a proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - h, c + h
