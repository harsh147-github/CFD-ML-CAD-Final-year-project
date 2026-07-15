"""
Module 1 - Step 2: Parametric ceiling-fan CAD builder (FreeCAD / OpenCASCADE).

Builds a WATERTIGHT solid of a full N-blade ceiling fan from a design vector,
and exports STEP + STL. Baseline defaults reproduce the reverse-engineered
GrabCAD blade (tip radius ~210 mm, hub ~25 mm, wide thin cambered paddle).

Rotation axis = global +Y (fan spins in the x-z plane), matching the baseline
STEP orientation so the OpenFOAM MRF zone (Module 2) is consistent.

Invocation (env-var driven, robust across FreeCADcmd arg quirks):
    set FAN_DESIGN_JSON=...design.json   (optional; else baseline)
    set FAN_OUT_STEP=...fan.step
    set FAN_OUT_STL=...fan.stl
    & FreeCADcmd blade_cad.py

design.json is a flat dict of the design variables below (any subset;
missing keys fall back to baseline).
"""
import json
import math
import os
import sys
import numpy as np

import Part
import FreeCAD as App
from FreeCAD import Base

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design_space as DS

# Fixed geometry (baseline size, held constant so every design is the same fan)
R_TIP = DS.FIXED["R_TIP_mm"]      # mm
R_HUB = DS.FIXED["R_HUB_mm"]      # mm
HUB_HEIGHT = DS.FIXED["HUB_HEIGHT_mm"]  # mm (axial height of hub cylinder, along Y)
N_SPAN = 16       # radial loft sections
N_CHORD = 60      # points around each airfoil

# Reference planform: normalized radius -> chord (mm), pre-scale.
# Fit to the reverse-engineered GrabCAD blade (one_blade_sections.py):
# narrow blade, wider bulbous root (~42 mm) settling to ~28-30 mm along span.
_PLANFORM_S = np.array([0.00, 0.15, 0.35, 0.60, 0.85, 1.00])
_PLANFORM_C = np.array([45.0, 40.0, 30.0, 29.0, 30.0, 28.0])


def load_design():
    d = DS.baseline()
    p = os.environ.get("FAN_DESIGN_JSON", "")
    if p and os.path.exists(p):
        with open(p) as f:
            user = json.load(f)
        d.update({k: user[k] for k in user if k in d})
    return d


# ---------------------------------------------------------------------------
# Thin cambered airfoil section (closed loop), chord along local x, camber in y.
# Uses NACA-4 camber line + small symmetric thickness -> watertight thin plate.
# ---------------------------------------------------------------------------
def airfoil_loop(n, camber, tc, camber_pos=0.4):
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))                      # 0..1 cosine spacing
    a0, a1, a2, a3, a4 = 0.2969, -0.1260, -0.3516, 0.2843, -0.1036  # closed TE
    yt = (tc / 0.2) * (a0*np.sqrt(x) + a1*x + a2*x**2 + a3*x**3 + a4*x**4)
    yc = np.zeros_like(x); dy = np.zeros_like(x)
    m = camber; p = camber_pos
    front = x <= p
    yc[front] = (m/p**2) * (2*p*x[front] - x[front]**2)
    dy[front] = (2*m/p**2) * (p - x[front])
    back = ~front
    yc[back] = (m/(1-p)**2) * ((1-2*p) + 2*p*x[back] - x[back]**2)
    dy[back] = (2*m/(1-p)**2) * (p - x[back])
    th = np.arctan(dy)
    xu = x - yt*np.sin(th); yu = yc + yt*np.cos(th)
    xl = x + yt*np.sin(th); yl = yc - yt*np.cos(th)
    xa = np.concatenate([xu, xl[::-1]])
    ya = np.concatenate([yu, yl[::-1]])
    return xa, ya   # in [0,1] chord units, camber in y


def chord_of(s, d):
    base = float(np.interp(s, _PLANFORM_S, _PLANFORM_C))
    # blend tip taper: scale the outer portion toward tip_taper*root-ref
    c = base * d["chord_scale"]
    # apply tip taper as an extra multiplier that ramps in over the outer half
    taper = 1.0 + (d["tip_taper"] - 1.0) * max(0.0, (s - 0.5) / 0.5)
    return c * taper


def pitch_of(s, d):
    return math.radians(d["global_pitch_deg"] + d["twist_deg"] * s)


def section_wire(r, d):
    """Build one closed airfoil wire at radius r (mm) as a FreeCAD Wire."""
    s = (r - R_HUB) / (R_TIP - R_HUB)
    c = chord_of(s, d)
    phi = pitch_of(s, d)
    xa, ya = airfoil_loop(N_CHORD, d["camber"], d["thickness_tc"])
    # center chord at mid-chord, scale to chord length
    u = c * (xa - 0.5)          # chordwise (tangential-ish)
    v = c * ya                  # camber/thickness (axial-ish)
    # rotate by pitch about the radial axis (in the tangential-axial plane)
    up = u*math.cos(phi) - v*math.sin(phi)
    vp = u*math.sin(phi) + v*math.cos(phi)
    # sweep: tangential offset grows with radius
    y_sweep = r * math.tan(math.radians(d["sweep_deg"]))
    # place into 3-D: radial along +X (blade 0), tangential along +Z, axial +Y
    pts = []
    for i in range(len(xa)):
        X = r
        Z = up[i] + y_sweep
        Y = vp[i]
        pts.append(Base.Vector(X, Y, Z))
    pts.append(pts[0])          # close
    return Part.makePolygon(pts)


def build_one_blade(d):
    # Start the root section INSIDE the hub cylinder (r < R_HUB) so the blade
    # overlaps the hub and the boolean fuse yields one watertight solid.
    r0 = 0.65 * R_HUB
    wires = [section_wire(r, d) for r in np.linspace(r0, R_TIP, N_SPAN)]
    loft = Part.makeLoft(wires, True, False, False)   # solid, not ruled, not closed
    return loft


def build_fan(d):
    blade = build_one_blade(d)
    n = int(d["n_blades"])
    yaxis = Base.Vector(0, 1, 0)
    origin = Base.Vector(0, 0, 0)
    solids = []
    for k in range(n):
        b = blade.copy()
        b.rotate(origin, yaxis, 360.0 * k / n)
        solids.append(b)
    # central hub cylinder along Y
    hub = Part.makeCylinder(R_HUB, HUB_HEIGHT, Base.Vector(0, -HUB_HEIGHT*0.5, 0), yaxis)
    solids.append(hub)
    # fuse into a single watertight solid
    fused = solids[0]
    for s in solids[1:]:
        fused = fused.fuse(s)
    fused = fused.removeSplitter()   # clean coplanar seams
    return fused


def main():
    d = load_design()
    print("Design:", json.dumps(d))
    fan = build_fan(d)
    print("Fan built: solids=%d shells=%d faces=%d volume=%.1f mm^3 valid=%s"
          % (len(fan.Solids), len(fan.Shells), len(fan.Faces),
             fan.Volume, fan.isValid()))
    bb = fan.BoundBox
    print("BBox mm: dx=%.1f dy=%.1f dz=%.1f" % (bb.XLength, bb.YLength, bb.ZLength))

    out_step = os.environ.get("FAN_OUT_STEP", "")
    out_stl = os.environ.get("FAN_OUT_STL", "")
    if out_step:
        fan.exportStep(out_step)
        print("Wrote STEP:", out_step)
    if out_stl:
        # meshed STL for CFD / visualization
        import Mesh
        import MeshPart
        doc = App.newDocument("fan")
        obj = doc.addObject("Part::Feature", "fan")
        obj.Shape = fan
        m = MeshPart.meshFromShape(Shape=fan, LinearDeflection=1.0,
                                   AngularDeflection=0.35, Relative=False)
        m.write(out_stl)
        print("Wrote STL:", out_stl, "facets=", m.CountFacets)

    # report
    rep = {
        "design": d, "valid": bool(fan.isValid()),
        "n_solids": len(fan.Solids), "n_faces": len(fan.Faces),
        "volume_mm3": float(fan.Volume),
        "bbox_mm": {"dx": bb.XLength, "dy": bb.YLength, "dz": bb.ZLength},
    }
    rp = os.environ.get("FAN_OUT_REPORT", "")
    if rp:
        with open(rp, "w") as f:
            json.dump(rep, f, indent=2)
        print("Wrote report:", rp)


main()
