"""Build CAD + metres-STL for the top surrogate-screened designs (sweep_pareto.csv)
so they can be CFD-confirmed. Names them sweep_15 (<=1.5x power budget) and
sweep_20 (<=2.0x). The <=1.0x screen is already covered by the doe_011 region."""
import os, sys, csv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "module1_cad"))
sys.path.insert(0, os.path.join(ROOT, "module2_openfoam"))
import design_space as DS
from fan_builder import build_design
from export_blade_stl import export as export_stl

RES = os.path.join(ROOT, "results")
STAGING = os.path.join(ROOT, "module2_openfoam", "staging")
LABELMAP = {"<=1.5x": "sweep_15", "<=2.0x": "sweep_20"}

with open(os.path.join(RES, "sweep_pareto.csv")) as f:
    for row in csv.DictReader(f):
        name = LABELMAP.get(row["budget"])
        if not name:
            continue
        design = {k: (int(float(row[k])) if k == "n_blades" else float(row[k]))
                  for k in DS.CONT_NAMES + ["n_blades"]}
        rep = build_design(design, os.path.join(RES, "geo", name), make_stl=True)
        export_stl(rep["stl_path"], os.path.join(STAGING, "blade_%s.stl" % name), scale=0.001)
        # stash the design for record step
        import json
        os.makedirs(os.path.join(RES, "al"), exist_ok=True)
        json.dump({"design": design, "pred_flow_cmm": float(row["pred_flow_cmm"]),
                   "pred_power_W": float(row["pred_power_W"]),
                   "geometry": {"volume_mm3": rep.get("volume_mm3"), "valid": rep.get("valid")}},
                  open(os.path.join(RES, "al", name + ".json"), "w"), indent=2)
        print("%s: valid=%s vol=%.0f pred_flow=%.2f" % (
            name, rep.get("valid"), rep.get("volume_mm3", 0), float(row["pred_flow_cmm"])))
