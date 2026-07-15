"""
Production PINN surrogate: an ENSEMBLE of physics-informed MLPs.

Why an ensemble: a single MLP gives no uncertainty. Training N nets from
different seeds gives (a) a better mean prediction and (b) an honest uncertainty
= the spread of the ensemble, which is LARGE exactly where data is sparse (the
high-airflow region) and small where it is dense. The optimizer uses that spread
so it never trusts an extrapolation without CFD confirmation.

Predicts non-dimensional (Cp, Cq) with target standardisation + soft
monotonicity physics priors, converts to dimensional power_W / flow_cmm.

Saves results/surrogate_pinn.pt (ensemble state dicts + scalers). Provides a
PINNEnsemble predictor with predict(X,omega) and predict_std(X,omega).

Usage:
    python surrogate_pinn.py --data ../results/dataset.csv --out ../results/surrogate_pinn --n 6
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "module1_cad"))
import design_space as DS

CONT = DS.CONT_NAMES
RHO = DS.OPERATING["rho"]
D = 2.0 * DS.FIXED["R_TIP_mm"] / 1000.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MONO = {"global_pitch_deg": (+1, 0), "chord_scale": (+1, +1), "camber": (+1, +1)}


class MLP(nn.Module):
    def __init__(self, n_in, h=24, depth=2):
        super().__init__()
        L = [nn.Linear(n_in, h), nn.SiLU()]
        for _ in range(depth - 1):
            L += [nn.Linear(h, h), nn.SiLU()]
        L += [nn.Linear(h, 2)]
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return self.net(x)


def _train_one(Xn, Ys, epochs, lam, seed):
    torch.manual_seed(seed)
    xt = torch.tensor(Xn, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(Ys, dtype=torch.float32, device=DEVICE)
    m = MLP(Xn.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=2e-4)
    for e in range(epochs):
        opt.zero_grad()
        loss = ((m(xt) - yt) ** 2).mean()
        if lam > 0 and e % 25 == 0:
            xc = torch.rand(128, Xn.shape[1], device=DEVICE, requires_grad=True)
            yc = m(xc); pen = 0.0
            for j, nm in enumerate(CONT):
                if nm in MONO:
                    for k in (0, 1):
                        s = MONO[nm][k]
                        if s:
                            g = torch.autograd.grad(yc[:, k].sum(), xc, create_graph=True)[0][:, j]
                            pen = pen + torch.relu(-s * g).mean()
            loss = loss + lam * pen
        loss.backward(); opt.step()
    return m


class PINNEnsemble:
    def __init__(self, models, lo, hi, ymu, ysd):
        self.models = models
        self.lo = np.asarray(lo); self.hi = np.asarray(hi)
        self.ymu = np.asarray(ymu); self.ysd = np.asarray(ysd)
        self.D = D; self.rho = RHO
        self.Xtrain = None    # normalised training inputs (set by train/load)

    def nearest_dist(self, X):
        """Min Euclidean distance (normalised space) from each X to training data."""
        if self.Xtrain is None or len(self.Xtrain) == 0:
            return np.zeros(len(np.atleast_2d(X)))
        Xn = (np.asarray(X, float) - self.lo) / (self.hi - self.lo + 1e-9)
        d = np.sqrt(((Xn[:, None, :] - self.Xtrain[None, :, :]) ** 2).sum(-1))
        return d.min(1)

    def _coeffs(self, X):
        Xn = (np.asarray(X, float) - self.lo) / (self.hi - self.lo + 1e-9)
        xt = torch.tensor(Xn, dtype=torch.float32, device=DEVICE)
        preds = []
        with torch.no_grad():
            for m in self.models:
                preds.append(m(xt).cpu().numpy() * self.ysd + self.ymu)
        P = np.stack(preds, 0)                 # (n_models, n, 2)
        return P.mean(0), P.std(0)

    def _eval(self, X, omega, return_std):
        mean, std = self._coeffs(X)
        power = mean[:, 0] * (self.rho * omega**3 * self.D**5)
        flow_cmm = mean[:, 1] * (omega * self.D**3) * 60.0
        if return_std:
            return (power, flow_cmm,
                    std[:, 0] * (self.rho * omega**3 * self.D**5),
                    std[:, 1] * (omega * self.D**3) * 60.0)
        return power, flow_cmm

    def predict(self, X, omega):
        return self._eval(X, omega, False)

    def predict_std(self, X, omega):
        return self._eval(X, omega, True)


def train_ensemble(df, n_models=6, epochs=3000, lam=0.03):
    omega = df["omega_rad_s"].values
    Cp = df["power_W"].values / (RHO * omega**3 * D**5)
    Cq = df["flow_m3s"].values / (omega * D**3)
    Y = np.stack([Cp, Cq], 1)
    X = df[CONT].values.astype(float)
    lo = np.array([DS.CONTINUOUS[k][1] for k in CONT])
    hi = np.array([DS.CONTINUOUS[k][2] for k in CONT])
    Xn = (X - lo) / (hi - lo)
    ymu, ysd = Y.mean(0), Y.std(0) + 1e-9
    Ys = (Y - ymu) / ysd
    models = [_train_one(Xn, Ys, epochs, lam, seed=s) for s in range(n_models)]
    ens = PINNEnsemble(models, lo, hi, ymu, ysd)
    ens.Xtrain = Xn
    return ens


def save(ens, path, Xtrain=None):
    torch.save({
        "state_dicts": [m.state_dict() for m in ens.models],
        "lo": ens.lo, "hi": ens.hi, "ymu": ens.ymu, "ysd": ens.ysd,
        "D": ens.D, "rho": ens.rho, "n_in": len(CONT), "cont": CONT,
        "Xtrain": Xtrain,   # normalised training inputs, for the trust-region penalty
    }, path)


def load(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    models = []
    for sd in ck["state_dicts"]:
        m = MLP(ck["n_in"]).to(DEVICE); m.load_state_dict(sd); m.eval()
        models.append(m)
    ens = PINNEnsemble(models, ck["lo"], ck["hi"], ck["ymu"], ck["ysd"])
    ens.Xtrain = ck.get("Xtrain")
    return ens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.path.join(RESULTS, "surrogate_pinn"))
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=3000)
    a = ap.parse_args()
    df = pd.read_csv(a.data)
    df = df[df["usable"] == True].reset_index(drop=True)
    if len(df) < 6:
        print("Need >=6 usable points; have %d." % len(df)); return
    ens = train_ensemble(df, n_models=a.n, epochs=a.epochs)
    save(ens, a.out + ".pt", Xtrain=ens.Xtrain)
    # sanity: report training-fit accuracy (not LOO — validate_surrogates does LOO)
    omega = df["omega_rad_s"].values
    p, q = ens.predict(df[CONT].values, omega)
    def mape(t, pr): return float(np.mean(np.abs((t - pr)/(np.abs(t)+1e-9)))*100)
    print("PINN ensemble (%d nets) trained on %d pts. Train-fit MAPE power=%.1f%% flow=%.1f%%"
          % (a.n, len(df), mape(df["power_W"].values, p), mape(df["flow_cmm"].values, q)))
    print("Saved ->", a.out + ".pt")


if __name__ == "__main__":
    main()
