"""
Module 3 driver: run OpenFOAM CFD over a batch of DOE designs to generate the
training dataset. Wraps Module 2's run_case.py, one design at a time (CFD is
CPU-bound; cases run serially, each using all cores).

Skips designs already solved (idempotent/restartable). After the batch, call
dataset_builder.py to aggregate.

Usage:
    python run_doe.py --doe ../results/doe.csv --n 8 --endtime 1500 --nproc 8 --viz
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
M2 = os.path.join(os.path.dirname(HERE), "module2_openfoam")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "module1_cad"))
sys.path.insert(0, M2)
import design_space as DS
import run_case

RESULTS = os.path.join(os.path.dirname(HERE), "results")


def load_doe(path):
    designs = []
    with open(path) as f:
        for row in csv.DictReader(f):
            d = {k: (int(row[k]) if k == "n_blades" else float(row[k]))
                 for k in DS.CONT_NAMES + ["n_blades"]}
            designs.append((row["design_id"], d))
    return designs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doe", default=os.path.join(RESULTS, "doe.csv"))
    ap.add_argument("--n", type=int, default=8, help="how many DOE designs to run")
    ap.add_argument("--endtime", type=int, default=1500)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--include-baseline", action="store_true")
    a = ap.parse_args()

    todo = []
    if a.include_baseline:
        todo.append(("baseline", DS.baseline()))
    todo += load_doe(a.doe)[:a.n]

    summary = []
    for name, design in todo:
        done = os.path.join(RESULTS, "cfd", name + ".json")
        if os.path.exists(done):
            with open(done) as f:
                prev = json.load(f)
            if prev.get("status") == "ok":
                print("[%s] already solved, skipping." % name)
                summary.append((name, prev.get("power_W"), prev.get("flow_cmm"),
                                prev.get("converged")))
                continue
        res = run_case.run(design, name, endtime=a.endtime, nproc=a.nproc, viz=a.viz)
        summary.append((name, res.get("power_W"), res.get("flow_cmm"),
                        res.get("converged"), res.get("status")))

    print("\n==== DOE BATCH SUMMARY ====")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
