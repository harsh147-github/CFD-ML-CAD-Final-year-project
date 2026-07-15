"""
The surrogate's actual job: evaluate a MASSIVE design sweep in milliseconds.

Generates N (default 1,000,000) Latin-Hypercube design vectors and runs the
trained PINN ensemble on ALL of them on the GPU — no CAD, no CFD — reporting the
throughput (designs/second). This is the whole value proposition: CFD is spent
only on the ~30 training points + a handful of confirmations, while the surrogate
screens the entire 7-D space.

Outputs:
  - throughput + timing (designs/s, and the CFD-equivalent wall-clock avoided)
  - the surrogate Pareto front over the N designs (max airflow per power level)
  - results/evidence/surrogate_sweep.png  (1e6 predicted points + CFD front overlaid)
  - results/sweep_pareto.csv              (top surrogate designs, for CFD confirmation)

HONEST framing: the surrogate over-predicts airflow in the sparse high-airflow
region (see surrogate_validation.json), so the sweep is a SCREEN — its top designs
are candidates to be CFD-confirmed, not final answers.

Usage:
    python surrogate_sweep.py --n 1000000
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
EVID = os.path.join(RESULTS, "evidence")
sys.path.insert(0, os.path.join(ROOT, "module1_cad"))
sys.path.insert(0, os.path.join(ROOT, "module4_pinn"))
import design_space as DS
import torch
from surrogate_pinn import load as load_pinn

CONT = DS.CONT_NAMES
OMEGA = DS.OPERATING["rpm"] * 2 * np.pi / 60.0


def batched_predict(ens, X, omega, batch=200000):
    """Predict power_W, flow_cmm for a large X in GPU batches."""
    P = np.empty(len(X)); Q = np.empty(len(X))
    for i in range(0, len(X), batch):
        p, q = ens.predict(X[i:i+batch], omega)
        P[i:i+batch] = p; Q[i:i+batch] = q
    return P, Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000000)
    a = ap.parse_args()
    os.makedirs(EVID, exist_ok=True)

    ens = load_pinn(os.path.join(RESULTS, "surrogate_pinn.pt"))
    lo = np.array([DS.CONTINUOUS[k][1] for k in CONT])
    hi = np.array([DS.CONTINUOUS[k][2] for k in CONT])

    # --- generate the massive design sweep ---
    t0 = time.time()
    X = qmc.scale(qmc.LatinHypercube(d=len(CONT), seed=7).random(n=a.n), lo, hi)
    t_gen = time.time() - t0
    print("Generated %s LHS designs in %.2fs (%s vars, 3 blades)" % (
        f"{a.n:,}", t_gen, len(CONT)), flush=True)

    # --- surrogate evaluation of ALL of them ---
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    P, Q = batched_predict(ens, X, OMEGA)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_eval = time.time() - t0
    rate = a.n / t_eval
    # a single OpenFOAM case here takes ~5-7 min; quote the avoided wall-clock
    cfd_days = a.n * 6.0 / 60.0 / 24.0    # if each were a 6-min CFD run
    print("PINN ensemble evaluated %s designs in %.3fs  ->  %s designs/s" % (
        f"{a.n:,}", t_eval, f"{rate:,.0f}"), flush=True)
    print("  (the same %s designs as CFD would take ~%.0f days at ~6 min each)" % (
        f"{a.n:,}", cfd_days), flush=True)
    import json as _json
    _json.dump({"n_designs": a.n, "eval_seconds": t_eval, "designs_per_second": rate,
                "cfd_equiv_days": cfd_days, "device": "cuda" if torch.cuda.is_available() else "cpu"},
               open(os.path.join(EVID, "surrogate_sweep.json"), "w"), indent=2)

    # --- surrogate Pareto front over the sweep ---
    order = np.argsort(P)                 # ascending power
    Pm = np.maximum.accumulate(Q[order])  # best flow seen up to each power level
    front_idx = order[np.r_[True, np.diff(Pm) > 0]]
    # top screened designs: max flow within a few power budgets
    base = pd.read_json(os.path.join(RESULTS, "cfd", "baseline.json"), typ="series")
    bpow = float(base["power_W"])
    screened = []
    for frac, label in ((1.0, "<=1.0x baseline power"), (1.5, "<=1.5x"), (2.0, "<=2.0x")):
        budget = bpow * frac
        mask = P <= budget
        if mask.any():
            j = np.argmax(np.where(mask, Q, -1))
            d = DS.vector_to_design(X[j], n_blades=3)
            screened.append({"budget": label, "pred_flow_cmm": float(Q[j]),
                             "pred_power_W": float(P[j]), **d})
    sdf = pd.DataFrame(screened)
    sdf.to_csv(os.path.join(RESULTS, "sweep_pareto.csv"), index=False)
    print("\nTop surrogate-screened designs (to be CFD-confirmed):", flush=True)
    for r in screened:
        print("  %-18s pred_flow=%.2f pred_power=%.4f pitch=%.1f twist=%.1f camber=%.3f chord=%.2f"
              % (r["budget"], r["pred_flow_cmm"], r["pred_power_W"],
                 r["global_pitch_deg"], r["twist_deg"], r["camber"], r["chord_scale"]), flush=True)

    # --- plot: 1e6 surrogate cloud + CFD-confirmed front ---
    fig, ax = plt.subplots(figsize=(8.5, 6))
    idx = np.random.RandomState(0).choice(a.n, size=min(40000, a.n), replace=False)
    ax.scatter(P[idx], Q[idx], s=2, c="lightsteelblue", alpha=0.25,
               label="%s surrogate designs (subsampled)" % f"{a.n:,}")
    fo = np.argsort(P[front_idx])
    ax.plot(P[front_idx][fo], Q[front_idx][fo], "b-", lw=1.5, label="surrogate Pareto front")
    # overlay the real CFD-confirmed designs
    try:
        cdf = pd.read_csv(os.path.join(RESULTS, "dataset.csv"))
        cdf = cdf[cdf["usable"] == True]
        ax.scatter(cdf["power_W"], cdf["flow_cmm"], s=28, c="tab:red",
                   edgecolor="k", lw=0.4, label="CFD-confirmed designs", zorder=5)
    except Exception:
        pass
    ax.set_xlabel("power input (W)"); ax.set_ylabel("air delivery (m³/min)")
    ax.set_title("Surrogate sweep of %s designs vs real CFD (350 rpm)" % f"{a.n:,}")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(EVID, "surrogate_sweep.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print("\nWrote", out, flush=True)
    print("Wrote", os.path.join(RESULTS, "sweep_pareto.csv"), flush=True)


if __name__ == "__main__":
    main()
