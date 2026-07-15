"""Generate a flat disk STL used as the fan air-delivery measurement surface.
The disk (radius ~ rotor radius) sits just below the fan on the rotation axis (Y);
integrating U.n over it gives the coherent downward throughflow (air delivery),
excluding the diffuse far-field recirculation that makes a full-plane integral
oscillate.
"""
import os
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "case_template", "constant", "geometry", "measure_disk.stl")


def make(radius=0.16, y=-0.10, n=96):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rim = np.column_stack([radius * np.cos(ang), np.full(n, y), radius * np.sin(ang)])
    center = np.array([[0.0, y, 0.0]])
    verts = np.vstack([center, rim])
    faces = [[0, i + 1, (i + 1) % n + 1] for i in range(n)]
    m = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    m.export(OUT, file_type="stl_ascii")
    print("Wrote", OUT, "radius=%.3f y=%.3f area=%.4f m2" % (radius, y, np.pi * radius**2))


if __name__ == "__main__":
    make()
