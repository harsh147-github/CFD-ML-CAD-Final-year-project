"""
Module 4 (companion): Gaussian-Process surrogate of the CFD.

GPs are far more sample-efficient than an MLP on the small, sparse datasets this
pipeline produces (tens of points in 7-D), and they return a calibrated
UNCERTAINTY (std) alongside the mean - exactly what the EGO active-learning loop
needs to decide where to spend the next real OpenFOAM run.

Predicts the non-dimensional coefficients Cp (power) and Cq (flow) with separate
GPs, then converts back to dimensional power_W / flow_cmm. Reports leave-one-out
MAPE so accuracy is honest.

Usage:
    python surrogate_gp.py --data ../results/dataset.csv --out ../results/surrogate_gp
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "module1_cad"))
import design_space as DS

CONT = DS.CONT_NAMES
RHO = DS.OPERATING["rho"]
D = 2.0 * DS.FIXED["R_TIP_mm"] / 1000.0


def nondim(df, omega):
    Cp = df["power_W"].values / (RHO * omega**3 * D**5)
    Cq = df["flow_m3s"].values / (omega * D**3)
    return Cp, Cq


def make_gp():
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=np.ones(len(CONT)), length_scale_bounds=(1e-2, 1e2), nu=2.5)
              + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e0)))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                    n_restarts_optimizer=8, alpha=1e-10)


class GPModel:
    """Two GPs (Cp, Cq) + normalisation, with dimensional predict(mean,std)."""
    def __init__(self, lo, hi):
        self.lo = np.asarray(lo); self.hi = np.asarray(hi)
        self.gp_cp = make_gp(); self.gp_cq = make_gp()
        self.D = D; self.rho = RHO

    def _n(self, X):
        return (np.asarray(X, float) - self.lo) / (self.hi - self.lo + 1e-9)

    def fit(self, X, Cp, Cq):
        Xn = self._n(X)
        self.gp_cp.fit(Xn, Cp); self.gp_cq.fit(Xn, Cq)
        return self

    def predict(self, X, omega, return_std=False):
        Xn = self._n(X)
        cp, cp_s = self.gp_cp.predict(Xn, return_std=True)
        cq, cq_s = self.gp_cq.predict(Xn, return_std=True)
        power = cp * (self.rho * omega**3 * self.D**5)
        flow_cmm = cq * (omega * self.D**3) * 60.0
        if return_std:
            p_s = cp_s * (self.rho * omega**3 * self.D**5)
            q_s = cq_s * (omega * self.D**3) * 60.0
            return power, flow_cmm, p_s, q_s
        return power, flow_cmm


def leave_one_out(X, Cp, Cq, lo, hi):
    n = len(X)
    pred = np.zeros((n, 2))
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        gp = GPModel(lo, hi).fit(X[m], Cp[m], Cq[m])
        Xn = gp._n(X[i:i+1])
        pred[i, 0] = gp.gp_cp.predict(Xn)[0]
        pred[i, 1] = gp.gp_cq.predict(Xn)[0]
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "results", "surrogate_gp"))
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    df = df[df.get("usable", True) == True].reset_index(drop=True)
    if len(df) < 5:
        print("Need >=5 usable points; have %d." % len(df)); return
    omega = df["omega_rad_s"].values
    Cp, Cq = nondim(df, omega)
    X = df[CONT].values.astype(float)
    lo = [DS.CONTINUOUS[k][1] for k in CONT]
    hi = [DS.CONTINUOUS[k][2] for k in CONT]

    print("GP surrogate on %d points (%d-D)" % (len(df), len(CONT)))
    loo = leave_one_out(X, Cp, Cq, lo, hi)
    P_loo = loo[:, 0] * (RHO * omega**3 * D**5)
    Q_loo = loo[:, 1] * (omega * D**3) * 60.0
    P_true, Q_true = df["power_W"].values, df["flow_cmm"].values
    sv_true = Q_true / np.maximum(P_true, 1e-9)
    sv_loo = Q_loo / np.maximum(P_loo, 1e-9)

    def mape(a_, b_):
        return float(np.mean(np.abs((a_ - b_) / (np.abs(a_) + 1e-9))) * 100)
    report = {"model": "GP", "n_points": len(df),
              "loo_mape_power_%": mape(P_true, P_loo),
              "loo_mape_flow_%": mape(Q_true, Q_loo),
              "loo_mape_service_value_%": mape(sv_true, sv_loo)}
    print(json.dumps(report, indent=2))

    model = GPModel(lo, hi).fit(X, Cp, Cq)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    # pickle the RAW sklearn GPs (always importable) not the wrapper class, so
    # the model unpickles cleanly in any process (avoids __main__.GPModel).
    with open(a.out + ".pkl", "wb") as f:
        pickle.dump({"gp_cp": model.gp_cp, "gp_cq": model.gp_cq,
                     "lo": np.asarray(lo), "hi": np.asarray(hi),
                     "D": D, "rho": RHO, "cont": CONT}, f)
    with open(a.out + "_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Saved GP surrogate ->", a.out + ".pkl")


if __name__ == "__main__":
    main()
