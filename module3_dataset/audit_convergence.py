"""
Convergence audit: for every solved case, report the honest convergence quality
(final residuals, torque coefficient-of-variation, flow coefficient-of-variation)
so we KNOW which data points are trustworthy before training or optimizing on them.

A case is trustworthy only if BOTH torque and flow are stable AND the momentum
residuals actually fell. Torque being stable while flow oscillates (opt_00) is a
common failure mode for lightly-loaded / unsteady designs and must be flagged.

Writes results/evidence/convergence_audit.csv and prints a summary.
"""
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
EVID = os.path.join(RESULTS, "evidence")
os.makedirs(EVID, exist_ok=True)
WSL = "//wsl.localhost/Ubuntu/home/harsh/fan_cfd"

RES_RE = re.compile(r"Solving for (U[xyz]|p),\s+Initial residual = ([-\deE.+]+)")


def final_residuals(case):
    """Return the mean of the last few initial-residuals for Ux and p."""
    log = os.path.join(WSL, os.path.basename(case), "log.foamRun")
    if not os.path.exists(log):
        return None, None
    ux, p = [], []
    try:
        with open(log, errors="ignore") as f:
            for line in f:
                m = RES_RE.search(line)
                if m:
                    if m.group(1) == "Ux":
                        ux.append(float(m.group(2)))
                    elif m.group(1) == "p":
                        p.append(float(m.group(2)))
    except OSError:
        return None, None
    fu = float(np.mean(ux[-10:])) if ux else None
    fp = float(np.mean(p[-10:])) if p else None
    return fu, fp


def main():
    rows = []
    for jp in sorted(glob.glob(os.path.join(RESULTS, "cfd", "*.json"))):
        with open(jp) as f:
            d = json.load(f)
        if d.get("status") != "ok":
            continue
        name = d["name"]
        fu, fp = final_residuals(os.path.join(WSL, name))
        tq_cov = d.get("torque_cov", np.nan)
        fl_cov = d.get("flow_cov", np.nan)
        # rigorous trustworthiness: both covs small AND residuals fell
        trust = (tq_cov < 0.05 and fl_cov < 0.05 and
                 (fu is not None and fu < 5e-3) and (fp is not None and fp < 5e-3))
        rows.append({
            "name": name,
            "power_W": round(d.get("power_W", 0), 5),
            "flow_cmm": round(d.get("flow_cmm", 0), 3),
            "SV": round(d.get("service_value_cmm_per_W", 0), 1),
            "torque_cov": round(tq_cov, 4),
            "flow_cov": round(fl_cov, 4),
            "res_Ux": None if fu is None else float("%.2e" % fu),
            "res_p": None if fp is None else float("%.2e" % fp),
            "converged_flag": d.get("converged"),
            "trustworthy": trust,
        })

    import csv
    out = os.path.join(EVID, "convergence_audit.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    ntrust = sum(r["trustworthy"] for r in rows)
    nflowbad = sum(r["flow_cov"] >= 0.05 for r in rows)
    print("=== CONVERGENCE AUDIT (%d cases) ===" % n)
    print("%-12s %9s %9s %9s %9s %9s %9s %s" %
          ("name", "power_W", "flow_cmm", "tq_cov", "fl_cov", "res_Ux", "res_p", "trust"))
    for r in sorted(rows, key=lambda x: x["flow_cov"]):
        print("%-12s %9.5f %9.3f %9.4f %9.4f %9.2e %9.2e %s" %
              (r["name"], r["power_W"], r["flow_cmm"], r["torque_cov"],
               r["flow_cov"], r["res_Ux"] or 0, r["res_p"] or 0,
               "OK" if r["trustworthy"] else "FLAG"))
    print("\nTrustworthy: %d/%d   |   flow oscillating (cov>=0.05): %d/%d" %
          (ntrust, n, nflowbad, n))
    print("Wrote", out)


if __name__ == "__main__":
    main()
