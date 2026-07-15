"""
Master driver: run the remaining pipeline automatically & resumably.

Phases (each idempotent; re-running skips finished work):
  A. wait for the detached doe2 CFD batch to finish
  B. parse every doe2 case -> results/cfd/doe2_XXX.json
  C. rebuild dataset + retrain GP surrogate
  D. active-learning optimization loop (train GP -> optimise -> ONE real
     OpenFOAM run at the candidate -> append -> retrain), until CFD agrees
     with the surrogate at the optimum or max iters
  E. export the CFD-confirmed best design to STEP/STL (FreeCAD) + screenshots
  F. generate the final report

Nothing is fabricated: the accepted optimum and every improvement are confirmed
by real OpenFOAM solves.
"""
import csv
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
PY = sys.executable
sys.path.insert(0, os.path.join(ROOT, "module1_cad"))
sys.path.insert(0, os.path.join(ROOT, "module2_openfoam"))
sys.path.insert(0, os.path.join(ROOT, "module4_pinn"))
sys.path.insert(0, os.path.join(ROOT, "module5_optimize"))
import design_space as DS
import parse_results
import run_case
from fan_builder import build_design

OMEGA = DS.OPERATING["rpm"] * 2 * 3.141592653589793 / 60.0


def log(msg):
    print("[full-pipeline] %s" % msg, flush=True)


# ---- Phase A: wait for detached doe2 batch ----
def wait_doe2(timeout_s=7200):
    prog = "/home/harsh/fan_cfd/batch_doe2.progress"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        r = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-c",
                            "grep -c 'batch complete' %s 2>/dev/null || echo 0" % prog],
                           capture_output=True, text=True)
        if r.stdout.strip().endswith("1") or "1" == r.stdout.strip():
            log("doe2 batch complete"); return True
        # also count solved cases for progress
        c = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-c",
                            "ls -d /home/harsh/fan_cfd/doe2_*/postProcessing 2>/dev/null | wc -l"],
                           capture_output=True, text=True)
        log("doe2 solved cases: %s" % c.stdout.strip())
        time.sleep(60)
    log("doe2 wait timed out; proceeding with whatever is done")
    return False


# ---- Phase B: parse doe2 cases -> jsons ----
def parse_doe2():
    doe2 = os.path.join(RESULTS, "doe2.csv")
    if not os.path.exists(doe2):
        return
    with open(doe2) as f:
        rows = {r["design_id"]: r for r in csv.DictReader(f)}
    for name, row in rows.items():
        jpath = os.path.join(RESULTS, "cfd", name + ".json")
        if os.path.exists(jpath):
            continue
        case = "//wsl.localhost/Ubuntu/home/harsh/fan_cfd/" + name
        if not os.path.isdir(os.path.join(case, "postProcessing", "rotorForces")):
            continue
        m = parse_results.analyze(case, OMEGA, tail_frac=0.3)
        if "error" in m:
            continue
        design = {k: (int(row[k]) if k == "n_blades" else float(row[k]))
                  for k in DS.CONT_NAMES + ["n_blades"]}
        rec = {"name": name, "design": design, "rpm": DS.OPERATING["rpm"],
               "omega_rad_s": OMEGA, "status": "ok"}
        rec.update(m)
        with open(jpath, "w") as f:
            json.dump(rec, f, indent=2)
        log("parsed %s: power=%.4f flow=%.3f SV=%.1f" %
            (name, rec["power_W"], rec["flow_cmm"], rec["service_value_cmm_per_W"]))


# ---- Phase C: dataset + surrogate ----
def rebuild_and_train():
    subprocess.run([PY, os.path.join(ROOT, "module3_dataset", "dataset_builder.py"),
                    "--in", os.path.join(RESULTS, "cfd"),
                    "--out", os.path.join(RESULTS, "dataset")], check=True)
    subprocess.run([PY, os.path.join(ROOT, "module4_pinn", "surrogate_gp.py"),
                    "--data", os.path.join(RESULTS, "dataset.csv"),
                    "--out", os.path.join(RESULTS, "surrogate_gp")], check=True)


# ---- Phase D: active-learning loop ----
def active_learning(max_iters=8, tol=0.15, patience=2, floor_frac=0.0):
    from optimize import Surrogate, optimize as surr_optimize
    history = []
    hpath = os.path.join(RESULTS, "optimization_history.json")
    if os.path.exists(hpath):
        history = json.load(open(hpath))
    start = len(history)
    good = 0
    for it in range(start, max_iters):
        log("=== active-learning iter %d ===" % it)
        rebuild_and_train()
        surr = Surrogate(os.path.join(RESULTS, "surrogate_gp.pkl"))
        # airflow floor = baseline flow (hold air delivery), read from baseline
        base = json.load(open(os.path.join(RESULTS, "cfd", "baseline.json")))
        floor = base["flow_cmm"] * 0.9
        cand = surr_optimize(surr, OMEGA, floor)
        log("candidate SV(pred)=%.1f pitch=%.1f camber=%.3f" % (
            cand["pred_service_value"], cand["design"]["global_pitch_deg"],
            cand["design"]["camber"]))
        name = "opt_%02d" % it
        res = run_case.run(cand["design"], name, endtime=800, nproc=8, viz=False)
        if res.get("status") != "ok":
            log("CFD failed at candidate: %s" % res.get("status")); break
        err = abs(res["service_value_cmm_per_W"] - cand["pred_service_value"]) / \
            max(cand["pred_service_value"], 1e-9)
        history.append({"iter": it, "candidate": cand,
                        "cfd": {"power_W": res["power_W"], "flow_cmm": res["flow_cmm"],
                                "service_value": res["service_value_cmm_per_W"],
                                "converged": res["converged"]}, "rel_err": err})
        json.dump(history, open(hpath, "w"), indent=2)
        log("iter %d: CFD SV=%.1f (pred %.1f, rel_err %.2f)" % (
            it, res["service_value_cmm_per_W"], cand["pred_service_value"], err))
        good = good + 1 if err < tol else 0
        if good >= patience:
            log("surrogate agrees with CFD at optimum -> converged"); break
    # pick best CFD-confirmed design
    if history:
        best = max(history, key=lambda r: r["cfd"]["service_value"])
        json.dump(best, open(os.path.join(RESULTS, "best_design.json"), "w"), indent=2)
        log("BEST CFD-confirmed SV=%.1f" % best["cfd"]["service_value"])
        return best
    return None


# ---- Phase E: export final CAD ----
def export_final(best):
    if not best:
        return
    design = best["candidate"]["design"]
    prefix = os.path.join(RESULTS, "optimized_fan")
    rep = build_design(design, prefix, make_stl=True)
    log("optimized CAD exported: %s.step valid=%s vol=%.0f" %
        (prefix, rep.get("valid"), rep.get("volume_mm3", 0)))
    # screenshots of geometry
    subprocess.run([PY, os.path.join(ROOT, "module1_cad", "render_geom.py"),
                    "--stl", prefix + ".stl",
                    "--out", os.path.join(RESULTS, "shots", "optimized"),
                    "--title", "Optimized fan"], check=False)
    # CFD screenshots of the confirmed optimum (its opt_NN case)
    name = "opt_%02d" % best["iter"]
    subprocess.run([PY, os.path.join(ROOT, "module2_openfoam", "visualize.py"),
                    "--case", "//wsl.localhost/Ubuntu/home/harsh/fan_cfd/" + name,
                    "--out", os.path.join(RESULTS, "cfd_shots", "optimized"),
                    "--title", "optimized"], check=False)


# ---- Phase F: report ----
def report():
    subprocess.run([PY, os.path.join(ROOT, "module6_validation", "generate_report.py"),
                    "--out", os.path.join(ROOT, "docs", "FINAL_REPORT.md")], check=True)


def main():
    log("start")
    wait_doe2()
    parse_doe2()
    best = active_learning()
    export_final(best)
    report()
    log("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
