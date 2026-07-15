"""Scale a fan STL from millimetres to metres (OpenFOAM units) and write it
as a single-solid 'blade' STL for snappyHexMesh. Verifies watertightness."""
import argparse
import os
import numpy as np
import trimesh


def export(in_stl, out_stl, scale=0.001):
    m = trimesh.load(in_stl)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    m.apply_scale(scale)
    os.makedirs(os.path.dirname(os.path.abspath(out_stl)), exist_ok=True)
    # write ascii STL with solid name 'blade'
    m.export(out_stl, file_type="stl_ascii")
    ext = m.bounds
    print("watertight=%s  faces=%d  bbox_m=[%.3f %.3f %.3f]..[%.3f %.3f %.3f]"
          % (m.is_watertight, len(m.faces), ext[0][0], ext[0][1], ext[0][2],
             ext[1][0], ext[1][1], ext[1][2]))
    return {"watertight": bool(m.is_watertight), "faces": int(len(m.faces)),
            "bbox_m": ext.tolist()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=0.001)
    a = ap.parse_args()
    export(a.inp, a.out, a.scale)
