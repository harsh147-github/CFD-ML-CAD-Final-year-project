"""
Render CFD screenshots for a solved fan case using pyvista (offscreen, GPU).
Reads the OpenFOAM case directly (VTK OpenFOAMReader) via a .foam stub.

Produces: pressure on blade, velocity-magnitude slice, pressure slice,
streamlines, and a combined montage. Fan rotation axis = Y.

Usage:
    python visualize.py --case //wsl.localhost/Ubuntu/home/harsh/fan_cfd/baseline --out ../results/cfd_shots/baseline --title baseline
"""
import argparse
import os
import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True


def load_case(case_dir):
    foam = os.path.join(case_dir, "case.foam")
    if not os.path.exists(foam):
        open(foam, "w").close()
    reader = pv.OpenFOAMReader(foam)
    try:
        reader.enable_all_patch_arrays()
    except Exception:
        pass
    reader.cell_to_point_creation = True
    times = reader.time_values
    if times:
        reader.set_active_time_value(times[-1])
    data = reader.read()
    return data, (times[-1] if times else None)


def get_blocks(data):
    internal = None
    boundary = None
    # MultiBlock: 'internalMesh' and 'boundary'
    for name in data.keys():
        blk = data[name]
        if name == "internalMesh":
            internal = blk
        elif name == "boundary":
            boundary = blk
    if internal is None:  # fallback: first UnstructuredGrid
        internal = data.combine()
    return internal, boundary


def find_blade(boundary):
    if boundary is None:
        return None
    for nm in boundary.keys():
        if "blade" in nm.lower():
            return boundary[nm]
    return None


def save_view(plotter, path):
    plotter.screenshot(path)
    print("Wrote", os.path.basename(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    data, t = load_case(a.case)
    internal, boundary = get_blocks(data)
    blade = find_blade(boundary)
    print("Loaded case at t=%s. internal cells=%d" % (t, internal.n_cells))
    arrs = list(internal.point_data.keys()) + list(internal.cell_data.keys())
    print("Fields:", sorted(set(arrs)))

    # velocity magnitude
    if "U" in internal.point_data:
        internal["Umag"] = np.linalg.norm(internal.point_data["U"], axis=1)

    # ---- 1. velocity magnitude slice (vertical plane z=0, shows axial jet) ----
    sl = internal.slice(normal=(0, 0, 1), origin=(0, 0, 0))
    p = pv.Plotter(off_screen=True, window_size=(1000, 800))
    p.add_mesh(sl, scalars="Umag", cmap="turbo", scalar_bar_args={"title": "|U| m/s"})
    if blade is not None:
        p.add_mesh(blade, color="black", opacity=0.3)
    p.add_text("%s  velocity |U|  (slice z=0)" % a.title, font_size=10)
    p.view_xy()  # look along -z? we want to see x-y plane
    p.camera_position = "xy"
    save_view(p, a.out + "_Umag_slice.png"); p.close()

    # ---- 2. pressure slice ----
    p = pv.Plotter(off_screen=True, window_size=(1000, 800))
    p.add_mesh(sl, scalars="p", cmap="coolwarm", scalar_bar_args={"title": "p m2/s2"})
    if blade is not None:
        p.add_mesh(blade, color="black", opacity=0.3)
    p.add_text("%s  pressure  (slice z=0)" % a.title, font_size=10)
    p.camera_position = "xy"
    save_view(p, a.out + "_p_slice.png"); p.close()

    # ---- 3. pressure on blade ----
    if blade is not None:
        p = pv.Plotter(off_screen=True, window_size=(1000, 800))
        p.add_mesh(blade, scalars="p", cmap="coolwarm",
                   scalar_bar_args={"title": "p m2/s2"})
        p.add_text("%s  blade surface pressure" % a.title, font_size=10)
        p.view_vector((0.4, 1, 0.4))
        save_view(p, a.out + "_blade_p.png"); p.close()

    # ---- 4. streamlines through the fan ----
    try:
        seed = pv.Disc(center=(0, 0.15, 0), inner=0.0, outer=0.16, normal=(0, 1, 0), r_res=6, c_res=30)
        strm = internal.streamlines_from_source(
            seed, vectors="U", max_time=2.0, integration_direction="both")
        p = pv.Plotter(off_screen=True, window_size=(1000, 800))
        p.add_mesh(strm.tube(radius=0.001), scalars="Umag" if "Umag" in strm.array_names else None, cmap="turbo")
        if blade is not None:
            p.add_mesh(blade, color="silver")
        p.add_text("%s  streamlines" % a.title, font_size=10)
        p.view_vector((1, 0.5, 0.6))
        save_view(p, a.out + "_streamlines.png"); p.close()
    except Exception as e:
        print("streamlines failed:", e)

    print("Visualization done.")


if __name__ == "__main__":
    main()
