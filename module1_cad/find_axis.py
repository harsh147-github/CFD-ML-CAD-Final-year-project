"""Find the TRUE rotation axis (hub center) and real blade radius of the
baseline GrabCAD fan, and count blades via angular histogram."""
import json, math, os
import numpy as np
import Part

HERE = os.path.dirname(os.path.abspath(__file__))
STEP = os.path.join(os.path.dirname(os.path.dirname(HERE)), "CAD Files", "GrabCAD_Ceiling_Fan.step")

shape = Part.Shape(); shape.read(STEP)
if shape.Solids:
    com = shape.Solids[0].CenterOfMass
else:
    com = shape.BoundBox.Center
print("CenterOfMass (mm): x=%.2f y=%.2f z=%.2f" % (com.x, com.y, com.z))

verts, tris = shape.tessellate(0.3)
P = np.array([[v.x, v.y, v.z] for v in verts])
x, y, z = P[:,0], P[:,1], P[:,2]

# candidate axis at center of mass (x,z), along Y
cx, cz = com.x, com.z
r = np.sqrt((x-cx)**2 + (z-cz)**2)
theta = np.degrees(np.arctan2(z-cz, x-cx))
print("About CoM axis: r_min=%.1f r_max=%.1f mm  (diameter=%.1f)" % (r.min(), r.max(), 2*r.max()))
print("Y extent: %.1f .. %.1f mm" % (y.min(), y.max()))

# blade count: histogram of angular position of the outer 30% of points
outer = r > 0.7*r.max()
hist, edges = np.histogram(theta[outer], bins=72, range=(-180,180))
# count peaks (lobes)
peaks = 0
for i in range(len(hist)):
    a,b,c = hist[i-1], hist[i], hist[(i+1)%len(hist)]
    if b>0 and b>=a and b>c and b>hist.mean():
        peaks += 1
print("Outer-point angular lobes (approx blade count):", peaks)
# print the angular distribution compactly
occ = "".join("#" if h>hist.mean() else "." for h in hist)
print("ang[-180..180]:", occ)

rep = {"com_mm":[com.x,com.y,com.z], "r_max_mm":float(r.max()),
       "r_min_mm":float(r.min()), "diameter_mm":float(2*r.max()),
       "y_min":float(y.min()),"y_max":float(y.max()),
       "approx_blades":int(peaks), "volume_mm3":float(shape.Volume)}
with open(os.path.join(HERE,"baseline_axis.json"),"w") as f: json.dump(rep,f,indent=2)
print("Wrote baseline_axis.json")
