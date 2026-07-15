"""
Module 1 - Step 3: Design-of-Experiments generator (Latin Hypercube Sampling).

Produces a space-filling sample of the continuous design space (plus a chosen
blade count) and writes doe.csv. The continuous space is 7-D; combined with the
3 discrete blade counts it spans the full design space the surrogate/optimizer
will search over millions of virtual designs.

Usage:
    python doe.py --n 20 --out ../results/doe.csv [--seed 1] [--nblades 3]
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design_space as DS


def make_doe(n, seed=1, n_blades=3, prefix="doe"):
    lows, highs = DS.bounds()
    d = len(lows)
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    unit = sampler.random(n=n)
    X = qmc.scale(unit, lows, highs)
    rows = []
    for i in range(n):
        design = DS.vector_to_design(X[i], n_blades=n_blades)
        design["design_id"] = "%s_%03d" % (prefix, i)
        rows.append(design)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--nblades", type=int, default=3)
    ap.add_argument("--prefix", default="doe")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results", "doe.csv"))
    args = ap.parse_args()

    rows = make_doe(args.n, seed=args.seed, n_blades=args.nblades, prefix=args.prefix)
    cols = ["design_id"] + DS.CONT_NAMES + ["n_blades"]
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})
    print("Wrote DOE: %s  (%d designs, %d continuous vars)" % (out, len(rows), len(DS.CONT_NAMES)))
    # quick sanity print
    print("Vars:", ", ".join(DS.CONT_NAMES))
    for r in rows[:3]:
        print(" ", r["design_id"], {k: round(r[k], 3) for k in DS.CONT_NAMES})


if __name__ == "__main__":
    main()
