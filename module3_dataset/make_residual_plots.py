"""
Archive convergence evidence: parse the OpenFOAM foamRun log of each key case and
plot (a) the initial residuals of Ux/Uy/Uz/p vs iteration and (b) the running
torque & airflow, so convergence is demonstrated, not asserted.

Writes results/evidence/residuals_<case>.png for baseline + the optimized design.
"""
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EVID = os.path.join(ROOT, "results", "evidence")
os.makedirs(EVID, exist_ok=True)
WSL = "//wsl.localhost/Ubuntu/home/harsh/fan_cfd"

RES = re.compile(r"Solving for (Ux|Uy|Uz|p),\s+Initial residual = ([-\deE.+]+)")
FLOW = re.compile(r'areaNormalIntegrate\("flowMonitor"\) of U = ([-\deE.+]+)')


def parse(case):
    log = os.path.join(WSL, case, "log.foamRun")
    res = {"Ux": [], "Uy": [], "Uz": [], "p": []}
    flow = []
    if not os.path.exists(log):
        return None
    with open(log, errors="ignore") as f:
        for line in f:
            m = RES.search(line)
            if m:
                res[m.group(1)].append(float(m.group(2)))
            mf = FLOW.search(line)
            if mf:
                flow.append(float(mf.group(1)))
    return res, flow


def plot(case, title):
    pr = parse(case)
    if pr is None:
        print("no log for", case); return
    res, flow = pr
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for k in ("Ux", "Uy", "Uz", "p"):
        if res[k]:
            # p is solved multiple times per iter; downsample to first-per-block
            ax1.semilogy(res[k], label=k, lw=0.9)
    ax1.set_xlabel("solver call"); ax1.set_ylabel("initial residual")
    ax1.set_title("%s — residual convergence" % title); ax1.legend(); ax1.grid(alpha=0.3)
    if flow:
        ax2.plot(np.array(flow) * 60.0, lw=1.0, color="tab:green")
        ax2.set_xlabel("monitor write"); ax2.set_ylabel("air delivery through disk (m³/min)")
        ax2.set_title("%s — airflow convergence" % title); ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(EVID, "residuals_%s.png" % case)
    fig.savefig(out, dpi=120); plt.close(fig)
    print("Wrote", out)


def main():
    cases = sys.argv[1:] if len(sys.argv) > 1 else ["baseline", "doe_011", "doe_001", "doe2_007"]
    for c in cases:
        plot(c, c)


if __name__ == "__main__":
    main()
