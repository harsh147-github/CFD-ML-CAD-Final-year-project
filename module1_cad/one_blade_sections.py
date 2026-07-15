"""Isolate ONE blade of the baseline fan (angular sector about the hub axis)
and extract its true chord / thickness / pitch distribution vs radius."""
import json, math, os
import numpy as np
import Part

HERE = os.path.dirname(os.path.abspath(__file__))
STEP = os.path.join(os.path.dirname(os.path.dirname(HERE)), "CAD Files", "GrabCAD_Ceiling_Fan.step")

shape = Part.Shape(); shape.read(STEP)
com = shape.Solids[0].CenterOfMass if shape.Solids else shape.BoundBox.Center
cx, cz = com.x, com.z
verts, _ = shape.tessellate(0.25)
P = np.array([[v.x, v.y, v.z] for v in verts])
x, y, z = P[:,0], P[:,1], P[:,2]
r = np.sqrt((x-cx)**2 + (z-cz)**2)
ang = np.arctan2(z-cz, x-cx)                       # radians
R = r.max()

# find blade lobe centers from outer points
outer = r > 0.6*R
hist, edges = np.histogram(ang[outer], bins=180, range=(-math.pi, math.pi))
centers = 0.5*(edges[:-1]+edges[1:])
# pick the single densest bin as lobe 0 center
lobe0 = centers[np.argmax(hist)]
print("Hub axis x=%.2f z=%.2f  R_tip=%.1f mm  lobe0=%.1f deg" % (cx, cz, R, math.degrees(lobe0)))

# select one blade: within +-55 deg of lobe0 (wrap-aware), r>0.12R (skip hub)
dang = np.arctan2(np.sin(ang-lobe0), np.cos(ang-lobe0))
sel = (np.abs(dang) < math.radians(55)) & (r > 0.12*R)
B = P[sel]; rB = r[sel]; angB = ang[sel]
print("Blade points:", len(B))

nb = 12
edges = np.linspace(rB.min(), R, nb+1)
rows = []
for i in range(nb):
    lo, hi = edges[i], edges[i+1]
    m = (rB>=lo) & (rB<=hi)
    if m.sum() < 6: continue
    band = B[m]
    rc = 0.5*(lo+hi)
    th = float(np.mean(angB[m]))
    tang = np.array([-math.sin(th), 0.0, math.cos(th)])   # tangential (chordwise)
    axial = np.array([0.0, 1.0, 0.0])                     # Y (axial)
    radial = np.array([math.cos(th), 0.0, math.sin(th)])
    c = band @ tang; a = band @ axial
    chord = float(c.max()-c.min())
    thick = float(a.max()-a.min())
    # geometric pitch: slope of chordline in (chordwise, axial) plane
    if c.max()-c.min() > 1e-6:
        A = np.polyfit(c, a, 1)
        pitch = math.degrees(math.atan(A[0]))
    else:
        pitch = 0.0
    rows.append({"r_mm":rc,"r_norm":(rc-rB.min())/(R-rB.min()),
                 "chord_mm":chord,"axial_env_mm":thick,"pitch_deg":pitch,"n":int(m.sum())})
    print("  r=%6.1f chord=%6.1f axialEnv=%5.1f pitch=%6.1f (n=%d)"
          % (rc, chord, thick, pitch, m.sum()))

rr = np.array([d["r_mm"] for d in rows]); ch = np.array([d["chord_mm"] for d in rows])
pt = np.array([d["pitch_deg"] for d in rows])
summary = {"hub_xz":[cx,cz],"R_tip_mm":float(R),"n_blades":3,
           "chord_root_mm":float(ch[0]),"chord_tip_mm":float(ch[-1]),
           "chord_mean_mm":float(ch.mean()),"chord_max_mm":float(ch.max()),
           "pitch_root_deg":float(pt[0]),"pitch_tip_deg":float(pt[-1]),
           "pitch_mean_deg":float(np.nanmean(pt))}
print("\nSUMMARY:", json.dumps(summary, indent=2))
with open(os.path.join(HERE,"one_blade_summary.json"),"w") as f:
    json.dump({"summary":summary,"sections":rows}, f, indent=2)
