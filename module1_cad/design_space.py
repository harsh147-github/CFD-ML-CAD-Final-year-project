"""
Canonical fan design space - single source of truth.

Pure-python (no third-party deps) so it imports cleanly under BOTH the system
Python (orchestration, DOE, ML) and FreeCAD's bundled Python (blade_cad.py).

Design variables derived by reverse-engineering the GrabCAD baseline blade
(reverse_engineer.py): wide thin cambered paddle, tip radius ~210 mm, hub ~25 mm.
"""

# name -> (baseline, low, high)
# Baselines calibrated to the reverse-engineered GrabCAD blade (narrow blade,
# tip radius ~148 mm, chord ~30-42 mm, thin cambered plate, pitch+washout).
CONTINUOUS = {
    "global_pitch_deg": (10.0, 4.0, 20.0),   # root pitch of the blade
    "twist_deg":        (-8.0, -18.0, 4.0),  # tip pitch relative to root (washout<0)
    "chord_scale":      (1.0, 0.75, 1.30),   # overall chord multiplier
    "camber":           (0.04, 0.01, 0.08),  # thin-plate max camber (fraction chord)
    "thickness_tc":     (0.04, 0.02, 0.08),  # section thickness / chord
    "sweep_deg":        (5.0, -10.0, 20.0),  # tangential skew with radius
    "tip_taper":        (0.90, 0.50, 1.20),  # tip chord / reference chord
}
DISCRETE = {"n_blades": (3, [3, 4, 5])}

# Fixed geometry (held constant so every design is the same-size fan).
# Measured from GrabCAD baseline: R_tip=147.7 mm, hub ~20 mm, 50 mm axial envelope.
FIXED = {
    "R_TIP_mm": 148.0,
    "R_HUB_mm": 20.0,
    "HUB_HEIGHT_mm": 24.0,
}

# Operating condition (fan rotational speed) - held for the base study.
OPERATING = {
    "rpm": 350.0,          # fan speed (omega = rpm*2pi/60)
    "rho": 1.2,            # kg/m^3 air density
    "nu": 1.5e-5,          # m^2/s kinematic viscosity
}

CONT_NAMES = list(CONTINUOUS.keys())


def baseline():
    d = {k: v[0] for k, v in CONTINUOUS.items()}
    d["n_blades"] = DISCRETE["n_blades"][0]
    return d


def bounds():
    """Return (lows, highs) lists for the continuous variables, in CONT_NAMES order."""
    lows = [CONTINUOUS[k][1] for k in CONT_NAMES]
    highs = [CONTINUOUS[k][2] for k in CONT_NAMES]
    return lows, highs


def vector_to_design(x, n_blades=3):
    """Map a continuous vector (CONT_NAMES order) + blade count to a design dict."""
    d = {CONT_NAMES[i]: float(x[i]) for i in range(len(CONT_NAMES))}
    d["n_blades"] = int(n_blades)
    return d


def design_to_vector(d):
    return [float(d[k]) for k in CONT_NAMES]
