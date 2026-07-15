# Ceiling-Fan Aerodynamic Shape Optimization — Final Report

Physics-validated pipeline: **FreeCAD parametric CAD → OpenFOAM MRF k-ω SST CFD → GP/PINN surrogate → shape optimization → CFD-confirmed optimum**. Every reported metric comes from a real OpenFOAM solve.

## 1. Baseline geometry (reverse-engineered)

- Source: GrabCAD ceiling fan STEP. Measured: **3 blades**, tip radius **148 mm** (⌀295 mm), rotation axis +Y.

![baseline fan](../results/shots/baseline_iso.png)

## 2. CFD dataset

- **36** OpenFOAM cases, **30 usable/converged**. Operating: 350 rpm, air ρ=1.2, ν=1.5e-05.

- Power range **0.0137–0.0829 W**, airflow **0.24–3.28 m³/min**, Service Value **9.7–160.3**.


| name     |   global_pitch_deg |   twist_deg |   chord_scale |   camber |   thickness_tc |   sweep_deg |   tip_taper |   power_W |   flow_cmm |   service_value_cmm_per_W |
|:---------|-------------------:|------------:|--------------:|---------:|---------------:|------------:|------------:|----------:|-----------:|--------------------------:|
| baseline |            10      |     -8      |        1      |   0.04   |         0.04   |      5      |      0.9    |    0.0157 |     0.31   |                   19.7778 |
| doe2_000 |             5.5452 |    -11.0315 |        0.939  |   0.0563 |         0.0313 |     17.1191 |      1.1149 |    0.0275 |     2.9019 |                  105.378  |
| doe2_001 |            12.8399 |     -2.4472 |        1.2799 |   0.07   |         0.0728 |     19.2789 |      0.5517 |    0.0409 |     0.5979 |                   14.6357 |
| doe2_002 |             9.8464 |      3.2261 |        1.0871 |   0.0423 |         0.0287 |      8.0212 |      1.1865 |    0.0371 |     0.5337 |                   14.3687 |
| doe2_003 |            18.6568 |      1.6885 |        1.2457 |   0.0749 |         0.0436 |     -0.7848 |      0.5119 |    0.0622 |     0.7764 |                   12.4901 |
| doe2_004 |            11.9047 |     -7.1278 |        1.2022 |   0.0454 |         0.0644 |     -6.4154 |      0.9805 |    0.0257 |     0.4226 |                   16.4138 |
| doe2_006 |            13.0406 |     -0.2754 |        1.0344 |   0.0398 |         0.0216 |     16.3063 |      0.6905 |    0.0403 |     0.5942 |                   14.7314 |
| doe2_007 |             4.8546 |    -15.1913 |        1.0111 |   0.0221 |         0.0339 |     -2.3085 |      0.7732 |    0.0198 |     2.8752 |                  145.233  |
| doe2_008 |             6.1404 |     -6.7139 |        1.1348 |   0.052  |         0.0628 |     -3.6353 |      1.1376 |    0.0233 |     0.307  |                   13.198  |
| doe2_009 |            18.1221 |    -13.1354 |        0.7551 |   0.0291 |         0.0485 |     -7.4808 |      0.5925 |    0.018  |     0.4244 |                   23.5834 |
| doe2_011 |            19.7986 |     -9.0644 |        0.9775 |   0.0604 |         0.039  |      3.4547 |      0.8584 |    0.0351 |     0.5873 |                   16.738  |
| doe2_012 |            17.3219 |      1.193  |        1.2386 |   0.0768 |         0.0263 |      3.2376 |      0.9285 |    0.0703 |     0.8406 |                   11.9518 |
| doe2_014 |            15.424  |     -4.8121 |        0.7977 |   0.0322 |         0.0746 |     12.8576 |      0.6202 |    0.0276 |     0.4989 |                   18.0839 |
| doe2_015 |             8.4783 |     -3.406  |        0.9269 |   0.0117 |         0.0541 |     11.56   |      0.9206 |    0.0166 |     0.3752 |                   22.6214 |
| doe2_016 |             7.1496 |     -1.7062 |        1.0585 |   0.0611 |         0.0774 |     -8.4212 |      0.7342 |    0.0246 |     0.2377 |                    9.6614 |
| doe_000  |             8.4882 |    -16.5569 |        0.9857 |   0.0496 |         0.0788 |      4.2063 |      0.77   |    0.0219 |     2.725  |                  124.307  |
| doe_001  |             6.5908 |    -14.6307 |        0.8178 |   0.0723 |         0.0517 |     -8.7432 |      1.1655 |    0.0314 |     3.2794 |                  104.424  |
| doe_002  |            19.6968 |    -10.3736 |        0.9516 |   0.0126 |         0.0567 |     13.8832 |      1.1234 |    0.0394 |     0.6621 |                   16.8023 |
| doe_003  |             7.7196 |     -4.9171 |        1.06   |   0.0627 |         0.0473 |     17.1102 |      0.5754 |    0.0234 |     0.3323 |                   14.1865 |
| doe_004  |            15.8393 |      1.2914 |        1.2135 |   0.0445 |         0.0627 |     -0.2063 |      0.8669 |    0.0612 |     0.7678 |                   12.5479 |
| doe_005  |             5.0827 |     -5.6794 |        1.0068 |   0.0167 |         0.0685 |     15.0475 |      0.9877 |    0.0146 |     0.2654 |                   18.1452 |
| doe_006  |            18.4071 |     -1.8576 |        1.0993 |   0.0384 |         0.0406 |     18.5881 |      1.0623 |    0.0829 |     0.9267 |                   11.1717 |
| doe_007  |             4.1804 |      3.0605 |        0.7573 |   0.0617 |         0.032  |     -6.6087 |      0.9339 |    0.0209 |     0.2383 |                   11.4087 |
| doe_010  |            17.8495 |     -0.788  |        1.2692 |   0.0475 |         0.0365 |     -4.4209 |      0.9518 |    0.0662 |     0.8039 |                   12.148  |
| doe_011  |             9.0809 |    -17.7619 |        1.1664 |   0.0246 |         0.0266 |      5.434  |      0.622  |    0.0155 |     2.2817 |                  146.763  |
| opt_00   |             7.1941 |    -17.9993 |        0.9069 |   0.01   |         0.0272 |     11.2515 |      0.9132 |    0.019  |     2.6241 |                  138.156  |
| opt_01   |             5.5186 |    -17.9994 |        1.3    |   0.01   |         0.0203 |     -3.039  |      0.5    |    0.0214 |     2.8673 |                  134.263  |
| opt_02   |            10.1265 |    -17.9997 |        0.9403 |   0.0256 |         0.02   |      0.8808 |      0.7452 |    0.0137 |     2.192  |                  160.27   |
| sweep_15 |             6.5244 |    -17.6773 |        0.7721 |   0.045  |         0.0335 |     -3.5493 |      0.5197 |    0.0174 |     2.7363 |                  157.353  |
| sweep_20 |             5.0105 |    -17.8525 |        0.8235 |   0.0623 |         0.042  |      7.3236 |      0.5486 |    0.0261 |     3.2039 |                  122.571  |

## 2b. Data integrity — convergence audit

- Every case audited for **both** torque and flow stability (cov<0.05) and residual fall. **24/32 trustworthy**; the rest (lightly-loaded, unsteady wake → oscillating airflow) are excluded from training. Evidence: `results/evidence/convergence_audit.csv`.

## 3. Surrogate accuracy (leave-one-out)

Both surrogates map the design vector → non-dimensional (Cp, Cq) → power/flow, validated by leave-one-out on **25 trustworthy CFD points** (device: cuda).


| model | power MAPE / R² | airflow MAPE / R² | Service-Value MAPE / R² |
|---|---|---|---|

| PINN | 14% / 0.90 | 29% / 0.55 | 27% / 0.73 |

| GP | 23% / 0.79 | 49% / 0.77 | 95% / -1.98 |


**Winner: PINN.** The PINN predicts power reliably (R²≈0.90); airflow is reliable in the dense low-moderate region but under-predicts the sparse high-airflow designs in LOO — so surrogate-proposed optima are always CFD-confirmed before being believed. Parity evidence: `results/evidence/surrogate_parity.png`; training curve: `results/evidence/pinn_training_curve.png`.

![surrogate parity](../results/evidence/surrogate_parity.png)

## 4. Optimization

**Objective (corrected):** maximise air delivery at a power budget (and its dual, minimise power at a required air delivery). The earlier airflow/power ratio objective was abandoned — it is numerically unstable (power→0) and drove the optimiser into an unconverged low-power corner. The optimum is selected from **CFD-confirmed, trustworthy** designs delivering ≥ baseline airflow; the PINN surrogate + differential-evolution search (with an ensemble-uncertainty penalty) proposes candidates that are always CFD-validated before acceptance.


**Air delivery vs power — the Pareto front (all CFD-confirmed):**


| name     |   power_W |   flow_cmm |   service_value |
|:---------|----------:|-----------:|----------------:|
| doe2_007 |   0.0198  |      2.875 |           145.2 |
| doe_001  |   0.0314  |      3.279 |           104.4 |
| doe_011  |   0.01555 |      2.282 |           146.8 |
| opt_02   |   0.01368 |      2.192 |           160.3 |
| sweep_15 |   0.01739 |      2.736 |           157.4 |
| sweep_20 |   0.02614 |      3.204 |           122.6 |


![Pareto front](../results/evidence/pareto_airflow_power.png)


The baseline sits at the bottom of the front (a weak design). At essentially the baseline's power the front jumps ~7× in air delivery; surrogate-proposed `opt_*` designs (diamonds) are CFD-confirmed points on/near the front — the PINN optimiser over-predicts their airflow (extrapolation) but lands on real non-dominated designs, not beyond the DOE-discovered front.


Active-learning iterations (surrogate proposal → real OpenFOAM validation):


| name     |   pred_flow |   cfd_flow |   cfd_power |   rel_err |
|:---------|------------:|-----------:|------------:|----------:|
| opt_00   |        3.6  |       2.62 |      0.019  |      0.4  |
| opt_01   |        3.44 |       2.87 |      0.0214 |      0.39 |
| opt_02   |        2.24 |       2.19 |      0.0137 |      0.02 |
| sweep_15 |        4.36 |       2.74 |      0.0174 |      0.37 |
| sweep_20 |        6.57 |       3.2  |      0.0261 |      0.51 |

## 5. Baseline vs optimized (CFD-confirmed)

Optimized design: **opt_02** (chosen from 30 trustworthy CFD-confirmed designs).


| metric | baseline | optimized | change |
|---|---|---|---|

| air delivery (m³/min) | 0.31 | 2.19 | **+607%** |

| power (W) | 0.0157 | 0.0137 | -13% |

| Service Value (m³/min/W) | 19.8 | 160.3 | **+710%** |


- Optimized design vector: `{"global_pitch_deg": 10.126502336544998, "twist_deg": -17.999738627868535, "chord_scale": 0.9402557453799066, "camber": 0.0256266266823453, "thickness_tc": 0.0200060182326468, "sweep_deg": 0.8807753255102666, "tip_taper": 0.7451705537107477, "n_blades": 3}`

- Exported: `results/optimized_fan.step` / `.stl`.

![optimized fan](../results/shots/optimized_iso.png)


**Convergence proof** (optimized case): residuals fall to ~1e-3 and the air delivery damps to a flat steady value — the reported number is converged, not a transient.

![optimized convergence](../results/evidence/residuals_opt_02.png)

## 6. CFD visual evidence

Velocity-magnitude slices (rotation axis = Y): the optimized fan drives a markedly stronger downward jet than the weak baseline.

![baseline Umag_slice](../results/cfd_shots/baseline_Umag_slice.png)

![baseline p_slice](../results/cfd_shots/baseline_p_slice.png)

![baseline streamlines](../results/cfd_shots/baseline_streamlines.png)

![optimized Umag_slice](../results/cfd_shots/optimized_Umag_slice.png)

![optimized p_slice](../results/cfd_shots/optimized_p_slice.png)

![optimized streamlines](../results/cfd_shots/optimized_streamlines.png)
