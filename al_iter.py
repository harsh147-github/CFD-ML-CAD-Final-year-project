"""
Active-learning helper (Windows python), called by the detached WSL orchestrator
via interop with Windows-path args. Three subcommands:

  propose  --name opt_NN : rebuild dataset -> retrain GP -> optimise Service Value
                           (airflow floor = 0.9x baseline) -> build candidate CAD
                           + export metres-STL to staging -> save al/<name>.json
  record   --name opt_NN : parse the solved WSL case -> results/cfd/<name>.json
  finalize               : pick best CFD-confirmed design -> export optimized CAD
                           + screenshots -> generate the final report

Kept quick (seconds–minutes) so it never trips the long-task killer; the CFD
itself runs in WSL. Nothing fabricated: designs come from the surrogate, metrics
from real OpenFOAM.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
PY = sys.executable
for m in ("module1_cad", "module2_openfoam", "module4_pinn", "module5_optimize"):
    sys.path.insert(0, os.path.join(ROOT, m))
import design_space as DS
import parse_results
from fan_builder import build_design
from export_blade_stl import export as export_stl

OMEGA = DS.OPERATING["rpm"] * 2 * 3.141592653589793 / 60.0
AL = os.path.join(RESULTS, "al")
STAGING = os.path.join(ROOT, "module2_openfoam", "staging")
os.makedirs(AL, exist_ok=True)


def _retrain():
    subprocess.run([PY, os.path.join(ROOT, "module3_dataset", "dataset_builder.py"),
                    "--in", os.path.join(RESULTS, "cfd"),
                    "--out", os.path.join(RESULTS, "dataset")], check=True)
    # PINN ensemble (validated winner: power R2 0.90, SV R2 0.73 vs GP's -1.98)
    subprocess.run([PY, os.path.join(ROOT, "module4_pinn", "surrogate_pinn.py"),
                    "--data", os.path.join(RESULTS, "dataset.csv"),
                    "--out", os.path.join(RESULTS, "surrogate_pinn"),
                    "--n", "5", "--epochs", "2000"], check=True)


def propose(name):
    _retrain()
    from optimize import Surrogate, max_flow_at_power
    surr = Surrogate(os.path.join(RESULTS, "surrogate_pinn.pt"))
    base = json.load(open(os.path.join(RESULTS, "cfd", "baseline.json")))
    # SOUND objective: deliver the most airflow within the baseline power budget,
    # with an uncertainty penalty keeping the candidate in the surrogate's trusted
    # region (avoids the unconverging low-power corner that the ratio objective hit).
    cand = max_flow_at_power(surr, OMEGA, base["power_W"], unc_weight=1.0)
    # build candidate CAD + metres STL
    rep = build_design(cand["design"], os.path.join(RESULTS, "geo", name), make_stl=True)
    export_stl(rep["stl_path"], os.path.join(STAGING, "blade_%s.stl" % name), scale=0.001)
    cand["geometry"] = {"volume_mm3": rep.get("volume_mm3"), "valid": rep.get("valid")}
    json.dump(cand, open(os.path.join(AL, name + ".json"), "w"), indent=2)
    print("PROPOSED %s: pred_flow=%.2f pred_power=%.4f pitch=%.2f twist=%.1f camber=%.3f flowstd=%.2f" % (
        name, cand["pred_flow_cmm"], cand["pred_power_W"],
        cand["design"]["global_pitch_deg"], cand["design"]["twist_deg"],
        cand["design"]["camber"], cand.get("pred_flow_std", 0)), flush=True)


def record(name):
    cand = json.load(open(os.path.join(AL, name + ".json")))
    case = "//wsl.localhost/Ubuntu/home/harsh/fan_cfd/" + name
    m = parse_results.analyze(case, OMEGA, tail_frac=0.3)
    if "error" in m:
        print("RECORD %s FAILED: %s" % (name, m["error"]), flush=True); return
    # convergence gate: an oscillating-flow candidate is NOT a valid data point.
    trust = (m.get("torque_cov", 1) < 0.05 and m.get("flow_cov", 1) < 0.05)
    rec = {"name": name, "design": cand["design"], "rpm": DS.OPERATING["rpm"],
           "omega_rad_s": OMEGA, "status": "ok",
           "geometry": cand.get("geometry", {}), "trustworthy_cfd": bool(trust)}
    rec.update(m)
    json.dump(rec, open(os.path.join(RESULTS, "cfd", name + ".json"), "w"), indent=2)
    # log to optimization history. Airflow is the primary target; use flow rel-err.
    hpath = os.path.join(RESULTS, "optimization_history.json")
    hist = json.load(open(hpath)) if os.path.exists(hpath) else []
    pred_flow = cand.get("pred_flow_cmm")
    flow_err = (abs(m["flow_cmm"] - pred_flow) / max(pred_flow, 1e-9)) if pred_flow else None
    hist.append({"iter": len(hist), "name": name, "candidate": cand,
                 "cfd": {"power_W": m["power_W"], "flow_cmm": m["flow_cmm"],
                         "service_value": m["service_value_cmm_per_W"],
                         "converged": m["converged"], "trustworthy": bool(trust)},
                 "rel_err": flow_err})
    json.dump(hist, open(hpath, "w"), indent=2)
    print("RECORDED %s: CFD flow=%.2f power=%.4f SV=%.1f (pred_flow %.2f, flow_rel_err %s) trust=%s" % (
        name, m["flow_cmm"], m["power_W"], m["service_value_cmm_per_W"],
        pred_flow or 0, ("%.2f" % flow_err) if flow_err is not None else "n/a", trust), flush=True)


def parse_doe2():
    """Parse any solved doe2 cases missing a json (called once before the loop)."""
    doe2 = os.path.join(RESULTS, "doe2.csv")
    if not os.path.exists(doe2):
        return
    import csv
    with open(doe2) as f:
        rows = {r["design_id"]: r for r in csv.DictReader(f)}
    for nm, row in rows.items():
        jp = os.path.join(RESULTS, "cfd", nm + ".json")
        if os.path.exists(jp):
            continue
        case = "//wsl.localhost/Ubuntu/home/harsh/fan_cfd/" + nm
        if not os.path.isdir(os.path.join(case, "postProcessing", "rotorForces")):
            continue
        m = parse_results.analyze(case, OMEGA, tail_frac=0.3)
        if "error" in m:
            continue
        d = {k: (int(row[k]) if k == "n_blades" else float(row[k]))
             for k in DS.CONT_NAMES + ["n_blades"]}
        rec = {"name": nm, "design": d, "rpm": DS.OPERATING["rpm"],
               "omega_rad_s": OMEGA, "status": "ok"}
        rec.update(m)
        json.dump(rec, open(jp, "w"), indent=2)
        print("parsed %s SV=%.1f" % (nm, m["service_value_cmm_per_W"]), flush=True)


def finalize():
    """Select the best CFD-confirmed, trustworthy design and build the honest
    baseline-vs-optimized comparison. Candidate pool = all usable (converged,
    non-oscillating) designs that deliver >= baseline airflow; winner = highest
    Service Value (air delivery per watt) among them. This is data-driven and
    every number is a real OpenFOAM solve."""
    import pandas as pd
    # rebuild the dataset first so the freshest CFD points (incl. the last
    # active-learning candidate) are in the selection pool.
    subprocess.run([PY, os.path.join(ROOT, "module3_dataset", "dataset_builder.py"),
                    "--in", os.path.join(RESULTS, "cfd"),
                    "--out", os.path.join(RESULTS, "dataset")], check=False)
    df = pd.read_csv(os.path.join(RESULTS, "dataset.csv"))
    df = df[df["usable"] == True].reset_index(drop=True)
    base = json.load(open(os.path.join(RESULTS, "cfd", "baseline.json")))
    bflow, bpower, bsv = base["flow_cmm"], base["power_W"], base["service_value_cmm_per_W"]

    # deliver at least baseline airflow, then maximise Service Value (efficiency)
    pool = df[df["flow_cmm"] >= bflow].copy()
    if len(pool) == 0:
        print("No trustworthy design meets the baseline airflow.", flush=True); return
    best = pool.loc[pool["service_value_cmm_per_W"].idxmax()]

    comp = {
        "baseline": {"name": "baseline", "power_W": bpower, "flow_cmm": bflow,
                     "service_value": bsv},
        "optimized": {"name": best["name"],
                      "power_W": float(best["power_W"]),
                      "flow_cmm": float(best["flow_cmm"]),
                      "service_value": float(best["service_value_cmm_per_W"]),
                      "design": {k: (int(best[k]) if k == "n_blades" else float(best[k]))
                                 for k in DS.CONT_NAMES + ["n_blades"]}},
        "improvement": {
            "airflow_pct": 100 * (best["flow_cmm"] - bflow) / bflow,
            "power_pct": 100 * (best["power_W"] - bpower) / bpower,
            "service_value_pct": 100 * (best["service_value_cmm_per_W"] - bsv) / bsv},
        "n_trustworthy_designs": int(len(df)),
    }
    json.dump(comp, open(os.path.join(RESULTS, "best_design.json"), "w"), indent=2)
    print("OPTIMIZED = %s: airflow %.2f->%.2f (%+.0f%%), power %.4f->%.4f (%+.0f%%), "
          "SV %.1f->%.1f (%+.0f%%)" % (
          best["name"], bflow, best["flow_cmm"], comp["improvement"]["airflow_pct"],
          bpower, best["power_W"], comp["improvement"]["power_pct"],
          bsv, best["service_value_cmm_per_W"], comp["improvement"]["service_value_pct"]),
          flush=True)

    # export the optimized geometry as CAD
    design = comp["optimized"]["design"]
    prefix = os.path.join(RESULTS, "optimized_fan")
    rep = build_design(design, prefix, make_stl=True)
    print("OPTIMIZED CAD: valid=%s vol=%.0f -> %s.step/.stl" % (
        rep.get("valid"), rep.get("volume_mm3", 0), prefix), flush=True)
    subprocess.run([PY, os.path.join(ROOT, "module1_cad", "render_geom.py"),
                    "--stl", prefix + ".stl",
                    "--out", os.path.join(RESULTS, "shots", "optimized"),
                    "--title", "Optimized fan"], check=False)
    subprocess.run([PY, os.path.join(ROOT, "module2_openfoam", "visualize.py"),
                    "--case", "//wsl.localhost/Ubuntu/home/harsh/fan_cfd/" + best["name"],
                    "--out", os.path.join(RESULTS, "cfd_shots", "optimized"),
                    "--title", "optimized"], check=False)
    subprocess.run([PY, os.path.join(ROOT, "module6_validation", "generate_report.py"),
                    "--out", os.path.join(ROOT, "docs", "FINAL_REPORT.md")], check=False)
    print("FINALIZED", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["propose", "record", "finalize", "parse_doe2"])
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    if a.cmd == "propose":
        propose(a.name)
    elif a.cmd == "record":
        record(a.name)
    elif a.cmd == "parse_doe2":
        parse_doe2()
    else:
        finalize()
