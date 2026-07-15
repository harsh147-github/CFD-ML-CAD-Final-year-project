"""
Module 5: Aerodynamic shape optimization over the surrogate, with EGO-style
active learning back to real OpenFOAM.

Inner loop (microseconds): differential_evolution over the trained surrogate
maximises Service Value (airflow per watt) subject to an airflow floor and the
design bounds. The surrogate evaluates millions of virtual blades with no CAD/CFD.

Active learning (outer loop): the proposed optimum is the next candidate to
confirm with a REAL OpenFOAM run (run_case.py). If CFD disagrees with the
surrogate, the point is added to the dataset and the surrogate retrained sharper
(Efficient Global Optimization). CAD/CFD are only ever spent here, deliberately.

This file provides the surrogate-side optimizer + the candidate proposer; the
driver loop (train -> optimize -> CFD-validate -> append -> repeat) is in
run_optimization.py.

Usage:
    python optimize.py --surrogate ../results/surrogate.pt --floor 0.5
"""
import argparse
import json
import os
import sys

import pickle

import numpy as np
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "module1_cad"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "module4_pinn"))
import design_space as DS
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "module4_pinn"))
from surrogate_pinn import load as load_pinn, CONT   # noqa: F401


class Surrogate:
    """Wraps the validated PINN ensemble (mean + ensemble-spread uncertainty).

    The PINN beat the GP on leave-one-out (power R2 0.90 vs 0.79, service-value
    R2 0.73 vs -1.98), so it is the production surrogate. `.pt` = PINN ensemble;
    a `.pkl` path still loads the GP for comparison."""
    def __init__(self, path):
        if path.endswith(".pt"):
            self._m = load_pinn(path); self._kind = "pinn"
        else:
            with open(path, "rb") as f:
                ck = pickle.load(f)
            self._gp = ck; self._kind = "gp"

    def predict(self, X, omega):
        if self._kind == "pinn":
            return self._m.predict(X, omega)
        return self._gp_eval(X, omega, False)

    def predict_std(self, X, omega):
        if self._kind == "pinn":
            return self._m.predict_std(X, omega)
        return self._gp_eval(X, omega, True)

    def _gp_eval(self, X, omega, return_std):
        ck = self._gp
        Xn = (np.asarray(X, float) - ck["lo"]) / (ck["hi"] - ck["lo"] + 1e-9)
        cp, cps = ck["gp_cp"].predict(Xn, return_std=True)
        cq, cqs = ck["gp_cq"].predict(Xn, return_std=True)
        power = cp * (ck["rho"] * omega**3 * ck["D"]**5)
        flow = cq * (omega * ck["D"]**3) * 60.0
        if return_std:
            return power, flow, cps * (ck["rho"]*omega**3*ck["D"]**5), cqs*(omega*ck["D"]**3)*60.0
        return power, flow


def _bounds():
    return [DS.CONTINUOUS[k][1] for k in CONT], [DS.CONTINUOUS[k][2] for k in CONT]


def _result(surr, omega, x):
    p, q, ps, qs = surr.predict_std([x], omega)
    return {"design": DS.vector_to_design(x, n_blades=3),
            "pred_power_W": float(p[0]), "pred_flow_cmm": float(q[0]),
            "pred_power_std": float(ps[0]), "pred_flow_std": float(qs[0]),
            "pred_service_value": float(q[0] / max(p[0], 1e-9))}


def _trust(surr, x, trust_radius, trust_weight):
    """Penalty (m3/min-equivalent) for straying beyond `trust_radius` of the
    nearest training point in normalised design space. Robust guard against the
    ensemble confidently extrapolating into unsampled corners."""
    if trust_weight <= 0 or not hasattr(surr, "_m") or getattr(surr._m, "Xtrain", None) is None:
        return 0.0
    d = float(surr._m.nearest_dist([x])[0])
    return trust_weight * max(0.0, d - trust_radius)


def max_flow_at_power(surr, omega, power_budget_W, unc_weight=1.0,
                      trust_radius=0.6, trust_weight=8.0, seed=0):
    """Maximise airflow s.t. predicted power <= budget. Airflow is the primary,
    numerically-stable objective. Two safeguards keep the optimiser honest rather
    than chasing extrapolated corners: (1) `unc_weight` * ensemble flow-std, and
    (2) a trust-region penalty for straying > `trust_radius` (normalised) from the
    nearest CFD-sampled design. Every proposal is still CFD-validated afterwards."""
    lo, hi = _bounds()

    def neg(x):
        p, q, ps, qs = surr.predict_std([x], omega)
        p, q, qs = float(p[0]), float(q[0]), float(qs[0])
        pen = max(0.0, p - power_budget_W) / max(power_budget_W, 1e-9) * 20.0
        return -(q - unc_weight * qs) + pen + _trust(surr, x, trust_radius, trust_weight)

    res = differential_evolution(neg, bounds=list(zip(lo, hi)), seed=seed,
                                 maxiter=200, tol=1e-7, polish=True, disp=False)
    r = _result(surr, omega, res.x)
    r["objective"] = "max_flow_at_power<=%.4fW" % power_budget_W
    r["dist_to_data"] = float(surr._m.nearest_dist([res.x])[0]) if hasattr(surr, "_m") else None
    return r


def min_power_at_flow(surr, omega, flow_target_cmm, unc_weight=1.0, seed=0):
    """Minimise power s.t. predicted airflow >= target ('same air, less watts')."""
    lo, hi = _bounds()

    def obj(x):
        p, q, ps, qs = surr.predict_std([x], omega)
        p, q, ps = float(p[0]), float(q[0]), float(ps[0])
        pen = max(0.0, flow_target_cmm - q) / max(flow_target_cmm, 1e-9) * 20.0
        return (p + unc_weight * ps) + pen * max(p, 1e-3)

    res = differential_evolution(obj, bounds=list(zip(lo, hi)), seed=seed,
                                 maxiter=200, tol=1e-7, polish=True, disp=False)
    r = _result(surr, omega, res.x)
    r["objective"] = "min_power_at_flow>=%.3fcmm" % flow_target_cmm
    return r


def pareto(surr, omega, n=12):
    """Sweep the airflow floor to trace the efficiency / air-delivery trade-off."""
    # first find the unconstrained max flow to set the sweep range
    lo = [DS.CONTINUOUS[k][1] for k in CONT]
    hi = [DS.CONTINUOUS[k][2] for k in CONT]
    def neg_flow(x):
        _, q = surr.predict([x], omega); return -float(q[0])
    r = differential_evolution(neg_flow, bounds=list(zip(lo, hi)), seed=1, maxiter=120)
    qmax = -r.fun
    floors = np.linspace(0.2 * qmax, 0.95 * qmax, n)
    front = []
    for fl in floors:
        front.append(optimize(surr, omega, fl))
    return front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogate", required=True)
    ap.add_argument("--floor", type=float, default=0.0, help="airflow floor cmm")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    omega = DS.OPERATING["rpm"] * 2 * np.pi / 60.0
    surr = Surrogate(a.surrogate)
    best = optimize(surr, omega, a.floor)
    print(json.dumps(best, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(best, f, indent=2)


if __name__ == "__main__":
    main()
