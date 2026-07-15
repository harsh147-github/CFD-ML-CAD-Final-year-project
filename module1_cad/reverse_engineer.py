"""
Module 1 - Step 1: Reverse-engineer the baseline ceiling-fan blade.

Run with:
    & "C:\\Program Files\\FreeCAD 1.0\\bin\\FreeCADcmd.exe" reverse_engineer.py

Method (fully geometry-driven, nothing assumed):
  1. Load the baseline STEP solid (one blade).
  2. Tessellate -> dense surface point cloud.
  3. The fan rotation axis is physically the global +Y (the thin ~50 mm
     direction). We measure radius r = sqrt(x^2 + z^2) about the Y axis
     through the origin, and the blade's radial span [r_root, r_tip].
  4. Slice the blade into radial bands. In each band we have an airfoil
     cross-section; measure:
        - chord    = max chord-wise extent in the x-z plane
        - thickness= extent perpendicular to chord within the band
        - pitch/twist = angle of the section's chord line w.r.t. the
          rotation plane (x-z), i.e. how much the section is tilted about
          the tangential direction (this is the aerodynamic twist).
  5. Fit simple distributions chord(r), twist(r), thickness(r) and dump
     everything to JSON + CSV so the parametric builder can reproduce the
     real baseline and perturb around it.

Every number is derived from the actual STEP file.
"""
import json
import math
import os
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(OUT_DIR))
CAD_DIR = os.path.join(PROJ, "CAD Files")
STEP_PATH = os.path.join(CAD_DIR, "GrabCAD_Ceiling_Fan.step")

REPORT = {}


def tessellate_points(shape, tol=0.5):
    """Return an (N,3) array of tessellation vertices of the solid surface."""
    verts, _tris = shape.tessellate(tol)
    pts = np.array([[v.x, v.y, v.z] for v in verts], dtype=float)
    return pts


def main():
    import Part
    print("Loading STEP:", STEP_PATH)
    shape = Part.Shape()
    shape.read(STEP_PATH)

    # true rotation axis = center of mass (x,z), along global Y
    com = shape.Solids[0].CenterOfMass if shape.Solids else shape.BoundBox.Center
    cx, cz = com.x, com.z
    print("Rotation axis at CoM: x=%.2f z=%.2f" % (cx, cz))

    pts = tessellate_points(shape, tol=0.4)
    print("Tessellated surface points:", len(pts))
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # --- rotation about Y axis through the hub center of mass ---
    r = np.sqrt((x - cx)**2 + (z - cz)**2)     # radius from hub axis
    theta = np.arctan2(z - cz, x - cx)         # angular position in x-z plane
    REPORT["rotation_center_xz_mm"] = [float(cx), float(cz)]
    r_root, r_tip = float(r.min()), float(r.max())
    print("Radius about Y axis: r_root=%.2f  r_tip=%.2f mm" % (r_root, r_tip))
    print("Y (axial/pitch) extent: %.2f .. %.2f mm" % (float(y.min()), float(y.max())))

    REPORT["rotation_axis"] = "Y (global vertical)"
    REPORT["units"] = "mm"
    REPORT["r_root_mm"] = r_root
    REPORT["r_tip_mm"] = r_tip
    REPORT["y_min_mm"] = float(y.min())
    REPORT["y_max_mm"] = float(y.max())
    REPORT["bbox_mm"] = {
        "dx": float(x.max() - x.min()),
        "dy": float(y.max() - y.min()),
        "dz": float(z.max() - z.min()),
    }
    REPORT["volume_mm3"] = float(shape.Volume)
    REPORT["n_surface_points"] = int(len(pts))

    # --- radial sectioning ---
    n_bands = 14
    edges = np.linspace(r_root, r_tip, n_bands + 1)
    sections = []
    for i in range(n_bands):
        lo, hi = edges[i], edges[i + 1]
        m = (r >= lo) & (r < hi if i < n_bands - 1 else r <= hi)
        if m.sum() < 8:
            continue
        band = pts[m]
        rc = 0.5 * (lo + hi)
        # --- per-band 3D PCA: robustly separate chord / span / thickness ---
        # PC1 = local chord direction (LE->TE), PC3 = thickness normal.
        cen = band.mean(axis=0)
        B = band - cen
        # covariance eigen-decomposition
        cov = np.cov(B.T)
        evals, evecs = np.linalg.eigh(cov)      # ascending eigenvalues
        order = np.argsort(evals)[::-1]         # descending
        pc1 = evecs[:, order[0]]                 # chord direction
        pc3 = evecs[:, order[2]]                 # thickness normal
        c1 = B @ pc1
        c3 = B @ pc3
        chord = float(c1.max() - c1.min())
        thick = float(c3.max() - c3.min())
        # geometric pitch = angle of chord line above the rotation plane (x-z).
        # |pc1_y| / |pc1_xz| -> tilt of the chord out of the horizontal plane.
        pc1_xz = math.hypot(pc1[0], pc1[2])
        pitch_deg = float(math.degrees(math.atan2(abs(pc1[1]), pc1_xz)))
        sections.append({
            "r_mm": rc,
            "r_norm": (rc - r_root) / (r_tip - r_root),
            "chord_mm": chord,
            "thickness_mm": thick,
            "thickness_ratio": thick / chord if chord > 1e-6 else 0.0,
            "pitch_deg": pitch_deg,
            "n_points": int(m.sum()),
        })

    REPORT["sections"] = sections
    print("\nRadial sections (r, chord, thick, t/c, pitch):")
    for s in sections:
        print("  r=%6.1f  chord=%6.1f  thick=%5.1f  t/c=%.3f  pitch=%6.1f  (n=%d)"
              % (s["r_mm"], s["chord_mm"], s["thickness_mm"],
                 s["thickness_ratio"], s["pitch_deg"], s["n_points"]))

    # --- summary distributions (for the parametric baseline) ---
    if sections:
        rr = np.array([s["r_mm"] for s in sections])
        ch = np.array([s["chord_mm"] for s in sections])
        tw = np.array([s["pitch_deg"] for s in sections])
        tc = np.array([s["thickness_ratio"] for s in sections])
        REPORT["baseline_summary"] = {
            "chord_root_mm": float(ch[0]),
            "chord_tip_mm": float(ch[-1]),
            "chord_max_mm": float(ch.max()),
            "chord_mean_mm": float(ch.mean()),
            "pitch_root_deg": float(tw[0]),
            "pitch_tip_deg": float(tw[-1]),
            "pitch_mean_deg": float(tw.mean()),
            "twist_root_to_tip_deg": float(tw[-1] - tw[0]),
            "tc_mean": float(tc.mean()),
            "tc_root": float(tc[0]),
            "tc_tip": float(tc[-1]),
        }
        print("\nBaseline summary:", json.dumps(REPORT["baseline_summary"], indent=2))

    # dump
    jp = os.path.join(OUT_DIR, "baseline_reverse_engineered.json")
    with open(jp, "w") as f:
        json.dump(REPORT, f, indent=2)
    cp = os.path.join(OUT_DIR, "baseline_sections.csv")
    with open(cp, "w") as f:
        f.write("r_mm,r_norm,chord_mm,thickness_mm,thickness_ratio,pitch_deg,n_points\n")
        for s in sections:
            f.write("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n" % (
                s["r_mm"], s["r_norm"], s["chord_mm"], s["thickness_mm"],
                s["thickness_ratio"], s["pitch_deg"], s["n_points"]))
    print("\nWrote:", jp)
    print("Wrote:", cp)


main()
