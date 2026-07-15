"""
Module 1 - Step 4: Plain-Python driver that builds ONE fan design by invoking
FreeCADcmd on blade_cad.py. Reused by the DOE runner (Module 3).

Returns the geometry report dict (valid, volume, bbox, faces).

Usage (standalone smoke test):
    python fan_builder.py --design ../results/doe.csv:doe_000 --out ../results/geo/doe_000
    python fan_builder.py --baseline --out ../results/geo/baseline
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import design_space as DS

FREECAD = os.environ.get(
    "FREECADCMD", r"C:\Program Files\FreeCAD 1.0\bin\FreeCADcmd.exe")
BLADE_CAD = os.path.join(HERE, "blade_cad.py")


def build_design(design, out_prefix, make_stl=True, timeout=300):
    """Build a fan for `design` dict. Writes <out_prefix>.step/.stl/.json.
    Returns the parsed geometry report dict (or raises on failure)."""
    out_prefix = os.path.abspath(out_prefix)
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    step = out_prefix + ".step"
    stl = (out_prefix + ".stl") if make_stl else ""
    report = out_prefix + ".json"

    dj = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(design, dj)
    dj.close()

    env = dict(os.environ)
    env["FAN_DESIGN_JSON"] = dj.name
    env["FAN_OUT_STEP"] = step
    env["FAN_OUT_STL"] = stl
    env["FAN_OUT_REPORT"] = report

    try:
        p = subprocess.run([FREECAD, BLADE_CAD], env=env, timeout=timeout,
                           capture_output=True, text=True)
    finally:
        try:
            os.unlink(dj.name)
        except OSError:
            pass

    if not os.path.exists(report):
        raise RuntimeError("FreeCAD build produced no report.\nSTDOUT:\n%s\nSTDERR:\n%s"
                           % (p.stdout[-2000:], p.stderr[-2000:]))
    with open(report) as f:
        rep = json.load(f)
    rep["step_path"] = step
    rep["stl_path"] = stl
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--design", help="path.csv:design_id  OR  path to a design json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-stl", action="store_true")
    args = ap.parse_args()

    if args.baseline or not args.design:
        design = DS.baseline()
    elif ".csv:" in args.design:
        path, did = args.design.rsplit(":", 1)
        import csv
        design = None
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["design_id"] == did:
                    design = {k: (int(row[k]) if k == "n_blades" else float(row[k]))
                              for k in DS.CONT_NAMES + ["n_blades"]}
                    break
        if design is None:
            raise SystemExit("design_id %s not found in %s" % (did, path))
    else:
        with open(args.design) as f:
            design = json.load(f)

    rep = build_design(design, args.out, make_stl=not args.no_stl)
    print(json.dumps({"valid": rep["valid"], "volume_mm3": round(rep["volume_mm3"], 1),
                      "n_faces": rep["n_faces"], "bbox_mm": rep["bbox_mm"],
                      "step": rep["step_path"]}, indent=2))


if __name__ == "__main__":
    main()
