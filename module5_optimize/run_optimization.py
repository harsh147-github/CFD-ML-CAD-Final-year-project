"""
Module 5 driver: closed-loop active-learning optimization (EGO).

Loop:
  1. Train the surrogate on the current CFD dataset (Module 4).
  2. Optimize Service Value over the surrogate (optimize.py) -> candidate x*.
  3. Run ONE real OpenFOAM CFD at x* (Module 2 run_case.py).
  4. Compare CFD vs surrogate at x*. Append the real point to the dataset.
  5. Repeat until the surrogate agrees with CFD at the optimum (|error| < tol)
     for `patience` consecutive iterations, or max_iters reached.

Real CAD/CFD is spent ONLY on the proposed optimum each round - the sample-
efficient core of the whole system. Nothing is fabricated: the accepted optimum
is confirmed by a real solve at the end (gate G4).

Usage:
    python run_optimization.py --max-iters 12 --endtime 1500 --nproc 8 --tol 0.05
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "module1_cad"))
sys.path.insert(0, os.path.join(ROOT, "module2_openfoam"))
sys.path.insert(0, os.path.join(ROOT, "module3_dataset"))
sys.path.insert(0, os.path.join(ROOT, "module4_pinn"))
import design_space as DS
import run_case
import dataset_builder
from optimize import Surrogate, optimize as surr_optimize

PY = sys.executable
RESULTS = os.path.join(ROOT, "results")


def rebuild_dataset():
    subprocess.run([PY, os.path.join(ROOT, "module3_dataset", "dataset_builder.py"),
                    "--in", os.path.join(RESULTS, "cfd"),
                    "--out", os.path.join(RESULTS, "dataset")], check=True)


def train_surrogate(epochs=4000):
    # GP surrogate: sample-efficient on sparse data + gives EGO uncertainty
    subprocess.run([PY, os.path.join(ROOT, "module4_pinn", "surrogate_gp.py"),
                    "--data", os.path.join(RESULTS, "dataset.csv"),
                    "--out", os.path.join(RESULTS, "surrogate_gp")], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iters", type=int, default=12)
    ap.add_argument("--endtime", type=int, default=1500)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--tol", type=float, default=0.05, help="rel surrogate-vs-CFD SV error to accept")
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--floor", type=float, default=0.0)
    a = ap.parse_args()
    omega = DS.OPERATING["rpm"] * 2 * np.pi / 60.0

    history = []
    good = 0
    for it in range(a.max_iters):
        print("\n===== ACTIVE-LEARNING ITERATION %d =====" % it)
        rebuild_dataset()
        train_surrogate()
        surr = Surrogate(os.path.join(RESULTS, "surrogate_gp.pkl"))
        cand = surr_optimize(surr, omega, a.floor)
        print("Surrogate optimum:", json.dumps(cand, indent=2))

        # confirm with a real CFD run
        name = "opt_%02d" % it
        res = run_case.run(cand["design"], name, endtime=a.endtime, nproc=a.nproc, viz=False)
        if res.get("status") != "ok":
            print("CFD failed at candidate; stopping.", res.get("status"))
            break
        cfd_sv = res.get("service_value_cmm_per_W", 0.0)
        err = abs(cfd_sv - cand["pred_service_value"]) / max(cand["pred_service_value"], 1e-9)
        rec = {"iter": it, "candidate": cand, "cfd": {
            "power_W": res["power_W"], "flow_cmm": res["flow_cmm"],
            "service_value": cfd_sv, "converged": res["converged"]}, "rel_err": err}
        history.append(rec)
        with open(os.path.join(RESULTS, "optimization_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        print("iter %d: surrogate SV=%.3f  CFD SV=%.3f  rel_err=%.3f" %
              (it, cand["pred_service_value"], cfd_sv, err))

        good = good + 1 if err < a.tol else 0
        if good >= a.patience:
            print("Converged: surrogate agrees with CFD at the optimum.")
            break

    print("\n==== OPTIMIZATION DONE ====")
    if history:
        best = max(history, key=lambda r: r["cfd"]["service_value"])
        print("Best CFD-confirmed design:", json.dumps(best, indent=2))
        with open(os.path.join(RESULTS, "best_design.json"), "w") as f:
            json.dump(best, f, indent=2)


if __name__ == "__main__":
    main()
