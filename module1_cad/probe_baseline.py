"""
Module 1 - Step 0: FreeCAD smoke test + baseline reverse-engineering probe.

Run with FreeCAD's bundled python:
    & "C:\\Program Files\\FreeCAD 1.0\\bin\\FreeCADcmd.exe" probe_baseline.py

Goals:
  1. Confirm FreeCAD Python (Part / Mesh / Import) actually loads and runs.
  2. Load the baseline GrabCAD ceiling-fan STEP, report overall geometry:
     bounding box, number of solids/shells, volume, per-solid bounding boxes.
  3. Load the baseline mesh (.msh via gmsh is separate) -> here we read the STL
     baseline as a cross-check on overall size.
  4. Write a JSON report of everything discovered so downstream parametrization
     can be pinned to the *real* fan dimensions (not guessed).

Nothing here is fabricated: every number printed comes from the actual file.
"""
import json
import os
import sys
import traceback

REPORT = {"freecad_ok": False, "step": {}, "stl": {}, "errors": []}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(OUT_DIR))  # ...\CFD ML CAD Major project
CAD_DIR = os.path.join(PROJ, "CAD Files")
STEP_PATH = os.path.join(CAD_DIR, "GrabCAD_Ceiling_Fan.step")
STL_PATH = os.path.join(CAD_DIR, "Ceiling Fan.stl")


def bbox_dict(bb):
    return {
        "xmin": bb.XMin, "xmax": bb.XMax,
        "ymin": bb.YMin, "ymax": bb.YMax,
        "zmin": bb.ZMin, "zmax": bb.ZMax,
        "dx": bb.XLength, "dy": bb.YLength, "dz": bb.ZLength,
        "diag": bb.DiagonalLength,
    }


def main():
    try:
        import FreeCAD  # noqa: F401
        import Part
        print("FreeCAD version:", ".".join(FreeCAD.Version()[:3]))
        REPORT["freecad_ok"] = True
        REPORT["freecad_version"] = ".".join(FreeCAD.Version()[:3])
    except Exception as e:
        REPORT["errors"].append("FreeCAD import failed: %s" % e)
        print("FATAL: FreeCAD import failed:", e)
        traceback.print_exc()
        _dump()
        sys.exit(2)

    # --- STEP ---
    try:
        import Part
        print("\nLoading STEP:", STEP_PATH)
        shape = Part.Shape()
        shape.read(STEP_PATH)
        bb = shape.BoundBox
        solids = shape.Solids
        shells = shape.Shells
        faces = shape.Faces
        print("  solids=%d shells=%d faces=%d" % (len(solids), len(shells), len(faces)))
        print("  overall bbox (mm): dx=%.2f dy=%.2f dz=%.2f diag=%.2f"
              % (bb.XLength, bb.YLength, bb.ZLength, bb.DiagonalLength))
        try:
            vol = shape.Volume
        except Exception:
            vol = None
        REPORT["step"] = {
            "path": STEP_PATH,
            "n_solids": len(solids),
            "n_shells": len(shells),
            "n_faces": len(faces),
            "bbox_mm": bbox_dict(bb),
            "volume_mm3": vol,
            "solids": [],
        }
        # per-solid: bbox + volume + centroid, so we can spot the blades/hub
        for i, s in enumerate(solids):
            sbb = s.BoundBox
            try:
                cx = s.CenterOfMass
                centroid = [cx.x, cx.y, cx.z]
            except Exception:
                centroid = None
            REPORT["step"]["solids"].append({
                "id": i,
                "volume_mm3": getattr(s, "Volume", None),
                "n_faces": len(s.Faces),
                "bbox_mm": bbox_dict(sbb),
                "centroid_mm": centroid,
            })
        # sort solids by volume for a quick view
        vs = sorted(REPORT["step"]["solids"], key=lambda d: (d["volume_mm3"] or 0), reverse=True)
        print("  largest solids by volume (mm^3):")
        for d in vs[:8]:
            print("    solid %d: vol=%.1f  dx=%.1f dy=%.1f dz=%.1f"
                  % (d["id"], d["volume_mm3"] or 0,
                     d["bbox_mm"]["dx"], d["bbox_mm"]["dy"], d["bbox_mm"]["dz"]))
    except Exception as e:
        REPORT["errors"].append("STEP load failed: %s" % e)
        print("STEP load failed:", e)
        traceback.print_exc()

    # --- STL cross-check ---
    try:
        import Mesh
        if os.path.exists(STL_PATH):
            print("\nLoading STL:", STL_PATH)
            m = Mesh.Mesh(STL_PATH)
            bb = m.BoundBox
            print("  facets=%d  bbox(mm): dx=%.2f dy=%.2f dz=%.2f"
                  % (m.CountFacets, bb.XLength, bb.YLength, bb.ZLength))
            REPORT["stl"] = {
                "path": STL_PATH,
                "n_facets": m.CountFacets,
                "bbox_mm": bbox_dict(bb),
            }
    except Exception as e:
        REPORT["errors"].append("STL load failed: %s" % e)
        print("STL load failed:", e)

    _dump()


def _dump():
    out = os.path.join(OUT_DIR, "baseline_probe.json")
    with open(out, "w") as f:
        json.dump(REPORT, f, indent=2)
    print("\nWrote report:", out)


# FreeCADcmd executes the script with __name__ != "__main__", so call directly.
main()
