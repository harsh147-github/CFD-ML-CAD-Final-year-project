"""Pre-build FreeCAD geometry + metres-STL for every design in a DOE csv, into
module2_openfoam/staging/blade_<id>.stl. Decouples the (Windows) CAD step from a
detached (WSL) mesh+solve loop, so the long CFD run survives harness task limits.
"""
import argparse, csv, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "module1_cad"))
sys.path.insert(0, os.path.join(ROOT, "module2_openfoam"))
import design_space as DS
from fan_builder import build_design
from export_blade_stl import export as export_stl

STAGING = os.path.join(ROOT, "module2_openfoam", "staging")
GEO = os.path.join(ROOT, "results", "geo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doe", required=True)
    a = ap.parse_args()
    with open(a.doe) as f:
        rows = list(csv.DictReader(f))
    ok = 0
    for row in rows:
        name = row["design_id"]
        stl_m = os.path.join(STAGING, "blade_%s.stl" % name)
        if os.path.exists(stl_m):
            print(name, "STL exists, skip"); ok += 1; continue
        design = {k: (int(row[k]) if k == "n_blades" else float(row[k]))
                  for k in DS.CONT_NAMES + ["n_blades"]}
        rep = build_design(design, os.path.join(GEO, name), make_stl=True)
        if not rep.get("valid"):
            print(name, "CAD INVALID"); continue
        export_stl(rep["stl_path"], stl_m, scale=0.001)
        print(name, "OK vol=%.0f" % rep["volume_mm3"]); ok += 1
    print("Pre-built %d/%d STLs -> %s" % (ok, len(rows), STAGING))


if __name__ == "__main__":
    main()
