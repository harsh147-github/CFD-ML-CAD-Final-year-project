"""
Module 2/3 orchestrator: run ONE full CFD case for a fan design, end to end.

  design dict --(Module 1 FreeCAD)--> STL(mm) --> STL(m) --> stage in WSL
  --> blockMesh + snappyHexMesh (MRF rotor cellZone) --> foamRun (kOmegaSST)
  --> parse torque/power/flow --> (optional) pyvista screenshots

All heavy lifting runs in WSL Ubuntu (OpenFOAM 12). Returns a metrics dict
and writes it to results/cfd/<name>.json. Nothing is fabricated: every metric
comes from the actual solver output.

Usage:
    python run_case.py --baseline --name baseline --endtime 2000 --nproc 8 --viz
    python run_case.py --design ../results/doe.csv:doe_003 --name doe_003 --endtime 1500 --nproc 8
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
M1 = os.path.join(os.path.dirname(HERE), "module1_cad")
sys.path.insert(0, M1)
sys.path.insert(0, HERE)
import design_space as DS
from fan_builder import build_design
from export_blade_stl import export as export_stl
import parse_results

RESULTS = os.path.join(os.path.dirname(HERE), "results")
STAGING = os.path.join(HERE, "staging")
SCRIPTS = os.path.join(HERE, "scripts").replace("\\", "/")
WIN_TO_MNT = lambda p: "/mnt/" + p[0].lower() + p[2:].replace("\\", "/")

STAGE_SH = "/mnt/f/CFD ML CAD Major project/fan_pipeline/module2_openfoam/scripts/stage_and_run.sh"
SOLVE_SH = "/mnt/f/CFD ML CAD Major project/fan_pipeline/module2_openfoam/scripts/solve.sh"
WSL_CASE = lambda name: "/home/harsh/fan_cfd/" + name
# forward-slash UNC (Windows accepts it; avoids backslash mangling through shells)
UNC_CASE = lambda name: "//wsl.localhost/Ubuntu/home/harsh/fan_cfd/" + name


def wsl(args, timeout=None, background=False):
    cmd = ["wsl", "-d", "Ubuntu", "--"] + args
    if background:
        return subprocess.Popen(cmd)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def run(design, name, endtime=2000, writeint=200, nproc=8, viz=False,
        omega=None, timeout=7200):
    t0 = time.time()
    log = {"name": name, "design": design}
    omega = omega if omega is not None else DS.OPERATING["rpm"] * 2 * 3.141592653589793 / 60.0
    log["omega_rad_s"] = omega
    log["rpm"] = DS.OPERATING["rpm"]

    # 1. build CAD (mm) then export STL in metres to staging/blade_<name>.stl
    print("[%s] building CAD..." % name)
    geo_prefix = os.path.join(RESULTS, "geo", name)
    rep = build_design(design, geo_prefix, make_stl=True)
    if not rep.get("valid"):
        return {**log, "status": "cad_invalid"}
    stl_mm = rep["stl_path"]
    stl_m = os.path.join(STAGING, "blade_%s.stl" % name)
    ex = export_stl(stl_mm, stl_m, scale=0.001)
    log["geometry"] = {"volume_mm3": rep["volume_mm3"], "watertight_m": ex["watertight"]}

    # 2. mesh + solve in WSL
    print("[%s] mesh + solve (endtime=%d nproc=%d)..." % (name, endtime, nproc))
    p = wsl(["bash", STAGE_SH, name, "%.4f" % omega, str(endtime), str(writeint),
             str(nproc), "run"], timeout=timeout)
    (open(os.path.join(RESULTS, "cfd", name + ".meshsolve.log"), "w")
     .write((p.stdout or "") + "\n---STDERR---\n" + (p.stderr or "")))
    if p.returncode != 0:
        return {**log, "status": "cfd_failed", "wsl_tail": (p.stdout or "")[-1500:] + (p.stderr or "")[-1500:]}

    # 3. parse metrics
    print("[%s] parsing results..." % name)
    metrics = parse_results.analyze(UNC_CASE(name), omega)
    log.update(metrics)
    log["status"] = "ok"
    log["wall_time_s"] = round(time.time() - t0, 1)

    os.makedirs(os.path.join(RESULTS, "cfd"), exist_ok=True)
    with open(os.path.join(RESULTS, "cfd", name + ".json"), "w") as f:
        json.dump(log, f, indent=2)

    # 4. optional visualization
    if viz and log.get("status") == "ok":
        try:
            subprocess.run([sys.executable, os.path.join(HERE, "visualize.py"),
                            "--case", UNC_CASE(name),
                            "--out", os.path.join(RESULTS, "cfd_shots", name),
                            "--title", name], timeout=600)
        except Exception as e:
            print("viz failed:", e)

    print("[%s] DONE: power=%.3f W flow=%.2f cmm SV=%.2f converged=%s" % (
        name, log.get("power_W", 0), log.get("flow_cmm", 0),
        log.get("service_value_cmm_per_W", 0), log.get("converged")))
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--design", help="path.csv:design_id or path.json")
    ap.add_argument("--name", required=True)
    ap.add_argument("--endtime", type=int, default=2000)
    ap.add_argument("--writeint", type=int, default=200)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--viz", action="store_true")
    a = ap.parse_args()

    if a.baseline or not a.design:
        design = DS.baseline()
    elif ".csv:" in a.design:
        import csv
        path, did = a.design.rsplit(":", 1)
        design = None
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["design_id"] == did:
                    design = {k: (int(row[k]) if k == "n_blades" else float(row[k]))
                              for k in DS.CONT_NAMES + ["n_blades"]}
        if design is None:
            raise SystemExit("design %s not found" % did)
    else:
        with open(a.design) as f:
            design = json.load(f)

    res = run(design, a.name, a.endtime, a.writeint, a.nproc, a.viz)
    print(json.dumps({k: res[k] for k in res if k not in ("design",)}, indent=2, default=str))


if __name__ == "__main__":
    main()
