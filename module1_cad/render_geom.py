"""
Module 1 - Step 5: Render an STL fan geometry to PNG screenshots (headless).

FreeCADcmd has no GUI, so we render the exported STL triangle mesh with
matplotlib 3-D (isometric + top + side views). Used to document every geometry.

Usage:
    python render_geom.py --stl ../results/baseline_fan.stl --out ../results/shots/baseline --title "Baseline fan"
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh


def render(stl_path, out_prefix, title=""):
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    mesh = trimesh.load(stl_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    tris = mesh.triangles  # (F,3,3)

    # face shading by normal.z for a bit of depth
    normals = mesh.face_normals
    shade = 0.4 + 0.6 * (0.5 * (normals[:, 1] + 1.0))  # light from +Y (axis)

    views = {"iso": (28, -60), "top": (89, -90), "side": (2, 0)}
    paths = []
    for name, (elev, azim) in views.items():
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        colors = np.zeros((len(tris), 4))
        colors[:, 0] = 0.20 * shade
        colors[:, 1] = 0.45 * shade
        colors[:, 2] = 0.85 * shade
        colors[:, 3] = 1.0
        pc = Poly3DCollection(tris, facecolors=colors, edgecolors=(0, 0, 0, 0.08),
                              linewidths=0.15)
        ax.add_collection3d(pc)
        v = mesh.vertices
        for setlim, arr in ((ax.set_xlim, v[:, 0]), (ax.set_ylim, v[:, 1]), (ax.set_zlim, v[:, 2])):
            setlim(arr.min(), arr.max())
        # equal aspect
        ext = np.ptp(v, axis=0)
        ax.set_box_aspect(ext + 1e-6)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y axis (mm)"); ax.set_zlabel("z (mm)")
        ax.set_title("%s  [%s]" % (title, name))
        p = "%s_%s.png" % (out_prefix, name)
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    print("Rendered:", ", ".join(os.path.basename(p) for p in paths))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    render(args.stl, args.out, args.title)


if __name__ == "__main__":
    main()
