"""
Module 3: aggregate all completed CFD cases into a traceable dataset.

Reads results/cfd/*.json (one per solved design, written by run_case.py) and
emits a tidy dataset (CSV + Parquet) with full traceability:
  design vars + operating condition + mesh/solver info + CFD outputs + metrics.

Only physically-valid, converged rows are marked usable=True; the integrity
rule (from the project spec) excludes non-converged / non-physical runs from
training while keeping them for audit.

Usage:
    python dataset_builder.py --in ../results/cfd --out ../results/dataset
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "module1_cad"))
import design_space as DS

CONT = DS.CONT_NAMES


def load_rows(indir):
    rows = []
    for jp in sorted(glob.glob(os.path.join(indir, "*.json"))):
        if jp.endswith(".meshsolve.log"):
            continue
        with open(jp) as f:
            d = json.load(f)
        if d.get("status") != "ok":
            continue
        design = d.get("design", {})
        row = {"name": d.get("name")}
        for k in CONT + ["n_blades"]:
            row[k] = design.get(k)
        row["rpm"] = d.get("rpm")
        row["omega_rad_s"] = d.get("omega_rad_s")
        # CFD outputs
        for k in ["torque_Nm", "power_W", "flow_m3s", "flow_cmm",
                  "service_value_cmm_per_W", "torque_cov", "flow_cov",
                  "plane_area_m2", "converged", "wall_time_s"]:
            row[k] = d.get(k)
        geo = d.get("geometry", {})
        row["volume_mm3"] = geo.get("volume_mm3")
        # physical-integrity checks
        phys = (row.get("power_W", 0) or 0) > 1e-3 and (row.get("flow_cmm", 0) or 0) > 1e-3
        row["physical"] = bool(phys)
        # STRICT trustworthiness: BOTH torque and flow must be stable (cov<0.05).
        # Torque-stable-but-flow-oscillating cases (lightly-loaded, unsteady wake)
        # give an unreliable airflow and must be excluded from training.
        tq = row.get("torque_cov")
        fl = row.get("flow_cov")
        trust = (tq is not None and fl is not None and tq < 0.05 and fl < 0.05)
        row["trustworthy"] = bool(trust)
        row["usable"] = bool(phys and trust)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default=os.path.join(os.path.dirname(HERE), "results", "cfd"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "results", "dataset"))
    a = ap.parse_args()

    rows = load_rows(a.indir)
    if not rows:
        print("No completed CFD cases found in", a.indir)
        return
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out + ".csv", index=False)
    try:
        df.to_parquet(a.out + ".parquet", index=False)
    except Exception as e:
        print("parquet skipped:", e)
    n_use = int(df["usable"].sum())
    print("Dataset: %d cases (%d usable) -> %s.csv" % (len(df), n_use, a.out))
    cols = ["name"] + CONT + ["n_blades", "power_W", "flow_cmm",
                                "service_value_cmm_per_W", "converged", "usable"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
