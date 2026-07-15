"""
Parse OpenFOAM postProcessing output for one fan case:
  - torque about the fan axis (Y) from rotorForces/.../forces.dat
  - net axial airflow from flowMonitor/.../surfaceFieldValue.dat
and derive power, air delivery, efficiency (Service Value = airflow/power).

Averages the last `tail_frac` fraction of iterations for a converged estimate,
and reports the coefficient of variation as a convergence indicator.

Usage:
    python parse_results.py --case //wsl.localhost/Ubuntu/home/harsh/fan_cfd/baseline --omega 36.65
"""
import argparse
import glob
import json
import os
import re
import numpy as np

VEC = r"\(([-\deE.+]+)\s+([-\deE.+]+)\s+([-\deE.+]+)\)"
# forces row: time ((Fp)(Fv)) ((Mp)(Mv))
ROW = re.compile(
    r"^\s*([-\deE.+]+)\s+"
    r"\(\s*" + VEC + r"\s*" + VEC + r"\s*\)\s*"
    r"\(\s*" + VEC + r"\s*" + VEC + r"\s*\)")


def _latest_dat(case, sub, fname):
    base = os.path.join(case, "postProcessing", sub)
    times = []
    if not os.path.isdir(base):
        return None
    for d in os.listdir(base):
        p = os.path.join(base, d, fname)
        if os.path.exists(p):
            try:
                times.append((float(d), p))
            except ValueError:
                pass
    if not times:
        return None
    times.sort()
    return times[-1][1]


def parse_forces(case):
    p = _latest_dat(case, "rotorForces", "forces.dat")
    if not p:
        return None
    torque_y = []
    times = []
    with open(p) as f:
        for line in f:
            if line.startswith("#"):
                continue
            m = ROW.match(line)
            if not m:
                continue
            g = [float(x) for x in m.groups()]
            t = g[0]
            Mp_y = g[1 + 6 + 1]   # time(1) + Fp(3)+Fv(3)=6, then Mp = g[7,8,9]
            Mv_y = g[1 + 6 + 3 + 1]
            times.append(t)
            torque_y.append(Mp_y + Mv_y)
    if not torque_y:
        return None
    return np.array(times), np.array(torque_y)


def parse_flow(case):
    p = _latest_dat(case, "flowMonitor", "surfaceFieldValue.dat")
    if not p:
        return None
    times = []
    vals = []
    area = None
    with open(p) as f:
        for line in f:
            if line.startswith("#"):
                m = re.search(r"Area\s*:\s*([-\deE.+]+)", line)
                if m:
                    area = float(m.group(1))
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    times.append(float(parts[0]))
                    vals.append(float(parts[-1]))
                except ValueError:
                    pass
    if not vals:
        return None
    return np.array(times), np.array(vals), area


def tail_stats(arr, frac=0.25):
    n = max(1, int(len(arr) * frac))
    tail = arr[-n:]
    mean = float(np.mean(tail))
    cov = float(np.std(tail) / (abs(mean) + 1e-12))
    return mean, cov


def analyze(case, omega, tail_frac=0.25):
    out = {"case": case, "omega_rad_s": omega, "converged": False}
    fr = parse_forces(case)
    fl = parse_flow(case)
    if fr is None or fl is None:
        out["error"] = "missing forces or flow output"
        return out
    _, tq = fr
    _, fv, area = fl
    torque, tq_cov = tail_stats(np.abs(tq), tail_frac)
    flow_m3s, fl_cov = tail_stats(fv, tail_frac)
    flow_m3s = abs(flow_m3s)
    power = omega * torque                 # W  (P = omega * torque)
    flow_cmm = flow_m3s * 60.0             # m^3/min
    sv = (flow_cmm / power) if power > 1e-9 else 0.0
    out.update({
        "torque_Nm": torque, "torque_cov": tq_cov,
        "power_W": power,
        "flow_m3s": flow_m3s, "flow_cmm": flow_cmm, "flow_cov": fl_cov,
        "plane_area_m2": area,
        "service_value_cmm_per_W": sv,
        "n_force_samples": int(len(tq)),
        "converged": bool(tq_cov < 0.15 and fl_cov < 0.10),
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--omega", type=float, required=True)
    ap.add_argument("--tail", type=float, default=0.25)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = analyze(a.case, a.omega, a.tail)
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
