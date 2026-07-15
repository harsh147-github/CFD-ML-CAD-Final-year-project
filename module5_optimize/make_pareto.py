"""
Pareto front of the real design trade-off: air delivery vs power input, over ALL
trustworthy CFD-confirmed designs. This is the honest representation — a fan is a
trade-off, not a single "optimum". Marks the baseline, the Pareto-optimal designs,
and the surrogate-proposed (opt_*) points so the reader sees whether the surrogate
optimiser found anything beyond the front.

Writes results/evidence/pareto_airflow_power.png and pareto_front.csv.
"""
import csv
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
EVID = os.path.join(RESULTS, "evidence")
os.makedirs(EVID, exist_ok=True)


def pareto_mask(power, flow):
    """A point is Pareto-optimal if no other point has >= flow AND <= power
    (strictly better in at least one)."""
    n = len(power); keep = np.ones(n, bool)
    for i in range(n):
        for j in range(n):
            if j != i and flow[j] >= flow[i] and power[j] <= power[i] and \
               (flow[j] > flow[i] or power[j] < power[i]):
                keep[i] = False; break
    return keep


def main():
    df = pd.read_csv(os.path.join(RESULTS, "dataset.csv"))
    df = df[df["usable"] == True].reset_index(drop=True)
    P = df["power_W"].values; Q = df["flow_cmm"].values
    names = df["name"].values
    is_opt = np.array([str(n).startswith("opt_") for n in names])
    is_base = names == "baseline"

    pm = pareto_mask(P, Q)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(P[~is_opt & ~is_base], Q[~is_opt & ~is_base], s=40, c="tab:blue",
               alpha=0.7, edgecolor="k", lw=0.4, label="DOE designs (CFD)")
    if is_opt.any():
        ax.scatter(P[is_opt], Q[is_opt], s=70, c="tab:orange", marker="D",
                   edgecolor="k", lw=0.5, label="surrogate-proposed (CFD)")
    ax.scatter(P[is_base], Q[is_base], s=130, c="red", marker="*",
               edgecolor="k", lw=0.6, label="baseline", zorder=5)
    # pareto front line
    order = np.argsort(P[pm])
    ax.plot(P[pm][order], Q[pm][order], "k--", lw=1.2, alpha=0.7, label="Pareto front")
    for i in np.where(pm)[0]:
        ax.annotate(names[i], (P[i], Q[i]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("power input (W)"); ax.set_ylabel("air delivery (m³/min)")
    ax.set_title("Air delivery vs power — %d trustworthy CFD designs (350 rpm)" % len(df))
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(EVID, "pareto_airflow_power.png")
    fig.savefig(out, dpi=130); plt.close(fig)

    # write the front
    with open(os.path.join(EVID, "pareto_front.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["name", "power_W", "flow_cmm", "service_value"])
        for i in np.where(pm)[0]:
            w.writerow([names[i], round(float(P[i]), 5), round(float(Q[i]), 3),
                        round(float(df["service_value_cmm_per_W"].values[i]), 1)])
    print("Pareto-optimal designs:")
    for i in np.where(pm)[0]:
        print("  %-10s power=%.4f W  flow=%.2f m3/min  SV=%.1f" % (
            names[i], P[i], Q[i], df["service_value_cmm_per_W"].values[i]))
    print("Wrote", out)


if __name__ == "__main__":
    main()
