"""
Module 4: Physics-informed surrogate of the CFD, trained on OpenFOAM data.

Maps the fan design vector -> (power_W, flow_cmm) with a small MLP on the GPU
(RTX 3060). Physics is embedded three ways so the model stays consistent on a
small dataset:
  1. Targets are learned as NON-DIMENSIONAL coefficients:
        power coefficient  Cp = power / (rho * omega^3 * D^5)
        flow  coefficient  Cq = flow_m3s / (omega * D^3)
     so the rho*omega^3*D^5 / omega*D^3 scalings are built in, not relearned.
  2. Soft MONOTONICITY penalties encode known trends (power rises with pitch,
     etc.) as a differentiable regulariser evaluated on random collocation pts.
  3. Leave-one-out cross-validation reports honest generalisation on few points.

This module intentionally works with as few as ~6-10 CFD points (like the
validated Crompton core) and improves as the active-learning loop adds data.

Usage:
    python surrogate.py --data ../results/dataset.csv --out ../results/surrogate
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
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "module1_cad"))
import design_space as DS

CONT = DS.CONT_NAMES
RHO = DS.OPERATING["rho"]
D = 2.0 * DS.FIXED["R_TIP_mm"] / 1000.0     # rotor diameter in metres

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def nondim(df, omega):
    """Return non-dimensional coefficients Cp, Cq from dimensional CFD outputs."""
    Cp = df["power_W"].values / (RHO * omega**3 * D**5)
    Cq = df["flow_m3s"].values / (omega * D**3)
    return Cp, Cq


class MLP(nn.Module):
    def __init__(self, n_in, n_hidden=32, n_out=2, depth=3):
        super().__init__()
        layers = [nn.Linear(n_in, n_hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.SiLU()]
        layers += [nn.Linear(n_hidden, n_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def normalize(X, lo, hi):
    return (X - lo) / (hi - lo + 1e-9)


# known monotone trends (sign of d(output)/d(var)); 0 = no strong prior.
# outputs: [Cp (power), Cq (flow)]
MONO = {
    "global_pitch_deg": (+1, +1),   # more pitch -> more power, more flow
    "chord_scale":      (+1, +1),   # bigger blades -> more power, more flow
    "camber":           (+1, +1),
}


def mono_penalty(model, lo, hi, n=256):
    """Differentiable monotonicity regulariser on random collocation points."""
    x = torch.rand(n, len(CONT), device=DEVICE, requires_grad=True)
    y = model(x)                      # already in normalised input space
    pen = 0.0
    for j, name in enumerate(CONT):
        if name not in MONO:
            continue
        for k in (0, 1):
            s = MONO[name][k]
            if s == 0:
                continue
            g = torch.autograd.grad(y[:, k].sum(), x, create_graph=True)[0][:, j]
            pen = pen + torch.relu(-s * g).mean()   # penalise wrong-sign slope
    return pen


def train(Xn, Y, lo, hi, epochs=4000, lam_mono=0.05, seed=0):
    torch.manual_seed(seed)
    model = MLP(Xn.shape[1]).to(DEVICE)
    xt = torch.tensor(Xn, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(Y, dtype=torch.float32, device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    for e in range(epochs):
        opt.zero_grad()
        pred = model(xt)
        loss = ((pred - yt) ** 2).mean()
        # monotonicity regulariser is expensive (create_graph autograd); apply
        # it periodically rather than every step.
        if lam_mono > 0 and e % 20 == 0:
            loss = loss + lam_mono * mono_penalty(model, lo, hi)
        loss.backward()
        opt.step()
    return model


def leave_one_out(Xn, Y, lo, hi, **kw):
    n = len(Xn)
    preds = np.zeros_like(Y)
    for i in range(n):
        mask = np.ones(n, bool); mask[i] = False
        m = train(Xn[mask], Y[mask], lo, hi, **kw)
        with torch.no_grad():
            preds[i] = m(torch.tensor(Xn[i:i+1], dtype=torch.float32,
                                      device=DEVICE)).cpu().numpy()[0]
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "results", "surrogate"))
    ap.add_argument("--epochs", type=int, default=4000)
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    df = df[df.get("usable", True) == True].reset_index(drop=True)
    if len(df) < 4:
        print("Need >=4 usable CFD points to train; have %d. Run more cases." % len(df))
        return
    omega = df["omega_rad_s"].values
    Cp, Cq = nondim(df, omega)
    Y = np.stack([Cp, Cq], axis=1)
    X = df[CONT].values.astype(float)
    lo = np.array([DS.CONTINUOUS[k][1] for k in CONT])
    hi = np.array([DS.CONTINUOUS[k][2] for k in CONT])
    Xn = normalize(X, lo, hi)

    print("Training on %d points, device=%s" % (len(df), DEVICE))
    # LOO validation
    loo = leave_one_out(Xn, Y, lo, hi, epochs=a.epochs)
    # back to dimensional for reporting
    def to_dim(Yc):
        P = Yc[:, 0] * (RHO * omega**3 * D**5)
        Q = Yc[:, 1] * (omega * D**3) * 60.0
        return P, Q
    P_true, Q_true = df["power_W"].values, df["flow_cmm"].values
    P_loo, Q_loo = to_dim(loo)
    sv_true = Q_true / np.maximum(P_true, 1e-9)
    sv_loo = Q_loo / np.maximum(P_loo, 1e-9)
    def mape(a_, b_):
        return float(np.mean(np.abs((a_ - b_) / (np.abs(a_) + 1e-9))) * 100)
    report = {
        "n_points": len(df), "device": DEVICE,
        "loo_mape_power_%": mape(P_true, P_loo),
        "loo_mape_flow_%": mape(Q_true, Q_loo),
        "loo_mape_service_value_%": mape(sv_true, sv_loo),
    }
    print(json.dumps(report, indent=2))

    # final model on all data
    model = train(Xn, Y, lo, hi, epochs=a.epochs)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "lo": lo, "hi": hi,
                "D": D, "rho": RHO, "cont": CONT}, a.out + ".pt")
    with open(a.out + "_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Saved surrogate ->", a.out + ".pt")


if __name__ == "__main__":
    main()
