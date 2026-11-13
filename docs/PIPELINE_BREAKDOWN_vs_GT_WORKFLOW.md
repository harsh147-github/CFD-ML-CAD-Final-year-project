# Pipeline breakdown and comparison with the GT workflow document

Scope: geometry ignored. Only the conceptual workflow is compared.
Compared: this repo (BLDC ceiling fan, `CFD-ML-CAD-Final-year-project`) vs. `GT_CSE_2027_SOP.pdf` (airframe).

**Method.** I read every source file in the repo except `reverse_engineer.py`, `probe_baseline.py`, `find_axis.py`, `one_blade_sections.py`, `render_geom.py`, `visualize.py`, `make_residual_plots.py` and `generate_report.py`. For those I read only the docstrings and header, because they are geometry or plotting code. I also checked the repo's own claims against `results/dataset.csv`, `results/al/*.json` and `results/cfd/*.json`.
**Not verified.** `torch` and `sklearn` are not installed in this environment, so I could not open `surrogate_pinn.pt` or `surrogate_gp.pkl`. OpenFOAM and FreeCAD logs live in WSL and are not in the repo. Nothing below was re-run. Every number comes from committed files.

---

## 1. The GT document, reduced to its conceptual steps

The PDF jumps from Step 2.2(A) straight to Step 5. It has no Steps 3–4 and no 2.2(B). Its numbers (500 LHS samples, 20 M cells, 25 POD modes, 12 PCA components, 8 GPUs) are illustrative targets, not results from an executed project.

| # | GT step | Core idea (geometry-free) |
|---|---|---|
| G1 | Sampling, AE 6373 | LHS over the design space; Kriging/GP response surface; **use predictive variance to pick follow-up CFD runs** |
| G2 | Geometry pipeline, CSE 6140 | Audit and repair the surface mesh (BVH, k-d tree, BFS normals, non-manifold edges) so it is simulation-ready |
| G3 | Output compression, CSE 6643 | SVD/POD of flow-field snapshots, giving ~25 modal coefficients per case |
| G4 | Input compression, CSE 6740 | PCA/Kernel-PCA of geometry, giving ~12 latent shape coordinates |
| G5 | Combine, "neural operator" | Learn map: latent shape coefficients → POD coefficients (field-level) |
| G6 | Deep architecture, CS 7643 | Transformer with multi-head attention; Adam; backprop; PyTorch |
| G7 | Physics, CSE 7752 | PINN/FNO with a **Navier–Stokes residual and continuity penalty in the loss** |
| G8 | UQ, ISYE 6420 | BNN + variational inference; NLL; aleatoric vs epistemic split; **high epistemic uncertainty triggers a background CFD run that is folded back into training** |
| G9 | Optimisation, AE 6310 | MDO; **adjoint gradients through the differentiable surrogate**; SQP; MDF architecture |
| G10 | Scale-out, CSE 6220 | MPI domain decomposition, custom CUDA, DDP, NCCL |

---

## 2. Your pipeline, atomised (what the code does)

Every stage lists input → operation → output → gate, with `file:line`.

### Stage 0. Configuration (single source of truth): `module1_cad/design_space.py`
1. 7 continuous variables, each `(baseline, lo, hi)` (`:14-22`): `global_pitch_deg` [4,20], `twist_deg` [−18,4], `chord_scale` [0.75,1.30], `camber` [0.01,0.08], `thickness_tc` [0.02,0.08], `sweep_deg` [−10,20], `tip_taper` [0.5,1.2].
2. 1 discrete variable, `n_blades` ∈ {3,4,5} (`:23`). **Never varied.** All 36 CFD cases have `n_blades=3`, `vector_to_design` defaults to 3 (`:56`), and the optimiser hard-codes 3 (`optimize.py:81`).
3. Fixed: R_tip 148 mm, R_hub 20 mm, hub height 24 mm (`:27-31`).
4. Operating point fixed: 350 rpm, ρ=1.2, ν=1.5e-5 (`:34-38`). **ω is the same in all 36 cases** (36.6519 rad/s).

### Stage 1. Baseline reverse engineering (`module1_cad/probe_baseline.py`, `find_axis.py`, `one_blade_sections.py`, `reverse_engineer.py`)
1. Load the GrabCAD STEP, tessellate it into a point cloud.
2. Find the rotation axis (solid centre of mass, global +Y); estimate blade count by angular histogram.
3. Slice radially, measure chord / thickness / pitch vs radius → `baseline_*.json/csv`.
4. These measurements calibrate `_PLANFORM_*` and the baseline values. The STEP file is **not committed** (`CAD Files/` is absent from the repo), so this stage can't be re-run from a clone.

### Stage 2. Parametric CAD (`module1_cad/blade_cad.py`, run through FreeCADcmd by `fan_builder.py`)
Per design vector:
1. `fan_builder.build_design` writes the design JSON to a temp file and sets env vars `FAN_DESIGN_JSON/OUT_STEP/OUT_STL/OUT_REPORT`.
2. It launches `FreeCADcmd.exe blade_cad.py` (300 s timeout).
3. `chord_of` (`:82`): interpolate the reference planform (6 knots), multiply by `chord_scale`, then by a taper ramp that starts at s=0.5.
4. `pitch_of` (`:91`): pitch(s) = `global_pitch + twist·s`, linear in span.
5. `airfoil_loop` (`:61`): NACA-4 camber line (max-camber position fixed at 0.4) plus a NACA-00xx thickness distribution, 60 points, closed trailing edge.
6. `section_wire` (`:95`): scale to chord, rotate by pitch, offset tangentially by `r·tan(sweep)`.
7. `build_one_blade` (`:120`): 16 radial sections (N_SPAN) from 0.65·R_hub (inside the hub) to R_tip → `Part.makeLoft` solid.
8. `build_fan` (`:129`): rotate N copies about +Y, fuse with a hub cylinder, `removeSplitter`.
9. Export STEP; mesh to STL (`MeshPart`, linear deflection 1.0 mm, angular 0.35); write a JSON report with `isValid()`, volume and bbox.
10. **Gate:** `run_case.py:63` aborts with `cad_invalid` if `valid` is false. Nothing else is gated at this stage.

### Stage 3. STL to OpenFOAM units (`module2_openfoam/export_blade_stl.py`)
1. `trimesh` loads the STL, scales mm→m (×0.001), writes ASCII STL named `blade`.
2. Prints `is_watertight`. **Logged only, not gated** (`run_case.py:68` stores it, nothing reads it).

### Stage 4. Design of experiments (`module1_cad/doe.py`)
1. `scipy.stats.qmc.LatinHypercube(d=7, seed=…)` → scale to bounds (`:27-29`).
2. Batch 1: `doe.csv`, seed 1, **16 designs generated, 12 run** (`doe_000–011`).
3. Batch 2: `doe2.csv`, seed 2, 18 designs, all run.
4. Plus 1 baseline run → 31 DOE-type points.
5. `prebuild_stls.py` pre-builds geometry → STL for a whole CSV, so CAD (Windows) is decoupled from the CFD loop (WSL).

### Stage 5. One CFD case (`module2_openfoam/run_case.py`, `scripts/*.sh`, `case_template/`)
1. `run_case.run` (`:51`): build CAD (Stage 2) → export STL (Stage 3) → `wsl bash stage_and_run.sh name ω endtime writeint nproc run` (`:72`).
2. `stage_and_run.sh`: copy template and STL into a space-free Linux path (`/home/harsh/fan_cfd/<name>`).
3. `setup_and_mesh.sh`: copy `0.orig/constant/system`; `sed`-substitute `__OMEGA__`, `__ENDTIME__`, `__WRITEINT__`, `__NPROC__`.
4. `blockMesh`: 0.9×1.2×0.9 m box, 36×48×36 cells (25 mm), one `farfield` patch.
5. `surfaceFeatures`, then `snappyHexMesh` **serially**:
   - blade surface level (3,4);
   - MRF zone = `searchableCylinder`, r=0.17 m, y ∈ [−0.06, 0.06], `cellZone rotor`, refinement level 2;
   - `addLayers false`, so there are **no prism layers**, only wall functions on the snapped mesh.
6. `checkMesh -constant … || true`. **Output is never parsed or gated.**
7. `solve.sh`: wipe old times, `0.orig`→`0`, `decomposePar` (scotch) → `mpirun -np 8 foamRun -parallel` → `reconstructPar`.
8. Physics (`case_template/`):
   - Solver `incompressibleFluid` (steady SIMPLE, SIMPLEC `consistent yes`).
   - `kOmegaSST`.
   - MRF about +Y at ω.
   - `U` blade wall = `MRFnoSlip`.
   - Farfield `pressureInletOutletVelocity` and `totalPressure p0=0`.
   - `linearUpwind` for U; `upwind` for k and ω.
   - `p`: GAMG, relTol 0.05; U/k/ω: smoothSolver, relTol 0.1.
   - Relaxation 0.7.
   - `residualControl` 1e-4.
   - Fixed `endTime` (iterations) is the real stop: 800 for doe2 and opt, 1200 for sweep, 1200–1500 for batch 1, 2000 default for baseline.
9. Function objects:
   - `forces` on patch `blade`, ρ_inf=1.2, CofR (0,0,0), every 5 iterations.
   - `surfaceFieldValue areaNormalIntegrate U` on `measure_disk.stl` (r=0.16 m, y=−0.10 m, 96-gon; `make_disk.py`).
10. `purgeWrite 2`: **only the last 2 time directories are kept, so no flow-field snapshot history exists.**

### Stage 6. Parse to two scalars (`module2_openfoam/parse_results.py`)
1. `parse_forces` (`:46`): regex over `forces.dat`; torque = Mp_y + Mv_y (pressure + viscous moment about +Y).
2. `parse_flow` (`:70`): last column of `surfaceFieldValue.dat` = net flux through the disk.
3. `tail_stats` (`:96`): mean and coefficient of variation (std/mean) over the last 25–30 % of the iteration history. The COV of a SIMPLE iteration series is an iteration-noise proxy, not a physical-time statistic.
4. `power = ω·|τ|`, `flow_cmm = |flux|·60`, `service_value = flow_cmm / power`.
5. `converged = torque_cov<0.15 and flow_cov<0.10` (`:126`). **A third, looser convergence definition** (see §6).

### Stage 7. Dataset assembly and gating (`module3_dataset/dataset_builder.py`, `audit_convergence.py`)
1. Load `results/cfd/*.json` (`status=="ok"`) → one row per case, 25 columns.
2. `physical` = power > 1e-3 W and flow > 1e-3 m³/min (`:53`).
3. `trustworthy` = torque_cov < 0.05 **and** flow_cov < 0.05 (`:60`).
4. `usable = physical & trustworthy` (`:62`). Current: **36 cases, 30 usable**.
5. `audit_convergence.py` is a *separate* audit that also requires final Ux and p residuals < 5e-3 (reads `log.foamRun`). Its output is **not** fed into the dataset.

### Stage 8. Surrogate training (`module4_pinn/surrogate_pinn.py`), the neural schema, see §3

### Stage 9. Surrogate validation (`module4_pinn/validate_surrogates.py`)
1. Load `usable` rows; non-dimensionalise; normalise inputs.
2. Leave-one-out for GP (Matern-2.5 ARD + White, 2 restarts) and for a **single** MLP (same architecture as one ensemble member, 2500 epochs).
3. Metrics: MAPE and R² for power, flow and service value.
4. Winner = lower **flow MAPE** (`:186`). The docstring says "by LOO service-value error, then flow error". The code and the docstring disagree.
5. Plots: parity, and a **training-loss** curve (no validation curve).
6. Result committed: n=25 (dataset now has 30 usable, so it is stale). PINN: power R² 0.90, flow R² 0.55, SV R² 0.73. GP: power 0.79, flow 0.77, SV −1.98.

### Stage 10. Optimiser (`module5_optimize/optimize.py`)
1. `Surrogate` wrapper loads the PINN `.pt` (or a GP `.pkl`).
2. `max_flow_at_power` (`:97`): `scipy.differential_evolution`, 7-D, `maxiter=200`, `polish=True`.
3. Objective (`:106-110`): minimise `−(q − 1.0·σ_q) + 20·max(0,P−budget)/budget + 8·max(0, d_nn − 0.6)`. That is:
   - flow with an uncertainty *subtraction* (pessimistic);
   - an exterior penalty for exceeding the power budget;
   - a trust-region penalty on nearest-training-point distance beyond 0.6 in normalised space.
4. Budget = baseline power (0.0157 W).
5. `min_power_at_flow` (`:120`): the dual; **not used anywhere**.
6. **`optimize()` is called but never defined** (see §6).

### Stage 11. Active-learning loop (`al_iter.py` driven by `scripts/al_orchestrator.sh`, detached in WSL)
1. Wait for the `doe2` batch to finish; `parse_doe2`.
2. For `it` in `0…NITERS−1` (fixed count; **no stopping rule**):
   - **propose** (`al_iter.py:50`): rebuild dataset → retrain PINN (**n=5 members, 2000 epochs**, `:47`) → `max_flow_at_power(budget=baseline power, unc_weight=1.0)` → build CAD → write `results/al/opt_NN.json`.
   - **solve:** `stage_and_run.sh opt_NN … endtime=800 … nproc=8`.
   - **record** (`al_iter.py:70`): parse → trust gate (both COVs < 0.05) → write `results/cfd/opt_NN.json` and append to `optimization_history.json`.
3. Executed: **3 iterations** (`opt_00–02`).
4. `finalize`: rebuild the dataset → pick the winner (Stage 13) → build the optimised CAD → render → `generate_report.py`.

### Stage 12. Surrogate sweep (`module5_optimize/surrogate_sweep.py`)
1. 1,000,000 LHS points (seed 7) → batched (200 k) GPU inference through the ensemble.
2. Surrogate Pareto front via sort-by-power + cumulative max of flow (`:96-98`).
3. For power budgets 1.0×/1.5×/2.0× baseline: pick `argmax(predicted flow)` within budget (`:103-110`). **Uses the ensemble mean only: no σ, no trust penalty.**
4. `build_sweep_cad.py` builds only the 1.5× and 2.0× picks → `sweep_15`, `sweep_20`.
5. `sweep_solve.sh` runs CFD at endtime 1200 after the AL loop finishes → `record`.

### Stage 13. Final selection (`al_iter.py:126`, `finalize`)
1. Pool = usable CFD cases with flow ≥ baseline flow.
2. "Optimised design" = `idxmax(service_value)` over the pool, **including DOE cases**. This is a post-hoc table lookup. It is not the optimiser's output. It uses the ratio objective the build log calls unstable.
3. Result: `opt_02` (it happens to be both the optimiser's proposal and the pool's best).

### Stage 14. Evidence and report (`make_pareto.py`, `make_residual_plots.py`, `visualize.py`, `generate_report.py`, `finalize_all.py`)
1. CFD-confirmed Pareto front (air vs power).
2. Residual-history plots, PyVista flow slices, report generation.
3. `finalize_all.py` orchestrates these. It contains **no surrogate-retrain step**, so the final `surrogate_pinn.pt` is whatever the last AL `propose` left behind.

### Dead and legacy code (present but not on the executed path)
- `module5_optimize/run_optimization.py` and `run_full_pipeline.py`: import the non-existent `optimize`; use the GP and the abandoned service-value objective. They fail at import.
- `module4_pinn/surrogate.py`: first-generation MLP (32×3, pitch prior (+1,+1)); `results/surrogate.pt` was trained on 10 points (LOO MAPE 88 / 122 / 84 %).
- `module5_optimize/optimize.py::pareto()`: calls undefined `optimize`.
- `module5_optimize/optimize.py::min_power_at_flow`: never called.

---

## 3. The neural schema, exact (production surrogate: `surrogate_pinn.py`)

| Item | Specification |
|---|---|
| Input | x ∈ ℝ⁷ = the 7 continuous design variables, min-max normalised by the design-space bounds to [0,1] (`+1e-9` in the denominator) |
| **Not inputs** | `n_blades` (constant 3), rpm (constant 350), any field, any geometry descriptor |
| Target | y = [C_p, C_q], with C_p = P / (ρ ω³ D⁵), C_q = Q / (ω D³), D = 0.296 m; then standardised (mean/std of the training set) |
| Network (per member) | `Linear(7,24) → SiLU → Linear(24,24) → SiLU → Linear(24,2)`; **842 parameters** |
| Data | **30 usable rows × 2 outputs = 60 target values** (14 parameters per target value) |
| Loss | `mean((ŷ − y)²)` full-batch **+ λ·P_mono**, λ=0.03 |
| P_mono | Every 25th epoch only (4 % of steps): 128 uniform random points in [0,1]⁷; `Σ mean(ReLU(−s · ∂ŷ_out/∂x_var))` over the prior set `MONO = {pitch: (C_p +1, C_q none), chord_scale: (+1,+1), camber: (+1,+1)}` |
| Optimiser | Adam, lr 3e-3, weight_decay 2e-4 (L2-style, not AdamW); full-batch; 3000 epochs CLI default, 2000 in the AL loop; no LR schedule, no early stopping, no validation split |
| Ensemble | N=6 (CLI default) or 5 (AL loop); seeds 0…N−1 change initialisation only. Same data, same architecture, no bootstrap |
| Prediction | mean over members → un-standardise → `P = C_p·ρω³D⁵`, `Q = C_q·ωD³·60` |
| Uncertainty | std over members (in coefficient space, rescaled the same way). No NLL, no aleatoric head, no calibration step |
| Trust metric | Nearest-neighbour Euclidean distance to training inputs in normalised space (`nearest_dist`) |
| Output type | **Two scalars.** No field, no mode coefficients, no spatial tokens |
| Comparator | GP: per-output `ConstantKernel × Matern(ν=2.5, ARD) + WhiteKernel`, `normalize_y`, 8 restarts, returns std |

Consequences of this schema:
- **The non-dimensionalisation does no work.** ω, ρ and D are constants in the dataset, so C_p and C_q are fixed rescalings of P and Q. The `ρω³D⁵` and `ωD³` scaling cannot be learned or exploited until rpm varies.
- **The only physics inside the network is the monotonicity prior.** The power priors are supported by the data (Spearman ρ with power: pitch 0.56, chord 0.44, camber 0.49). The two flow priors (chord +1, camber +1) are not supported: ρ(flow) is −0.04 for chord and −0.02 for camber. Two of the optimiser proposals (`opt_00`, `opt_01`) sit at the camber lower bound (0.01) with flow 2.6–2.9 m³/min, and the mean camber of the flow > 2 cases (0.038) is below that of the rest (0.046). The prior is soft, so it acts as a mild bias.
- Twist is the dominant flow driver (Spearman ρ = −0.56) and has no prior.

---

## 4. Step-by-step deviation map

Verdict key: **Partial** = same concept, reduced or altered. **Replaced** = a different mechanism meets the same goal. **Absent** = not implemented. **Different** = same goal, different algorithm class.

| GT step | What the GT document does | What you do | Verdict | Scale-driven (legit) or conceptual? |
|---|---|---|---|---|
| G1 LHS + Kriging | LHS (500 pts), GP response surface, **variance directs new CFD** | LHS ✔ (7-D, `qmc`); GP ✔ (Matern-2.5 ARD, n=30, LOO); variance **penalises** candidates rather than directing sampling (`q − σ`, no EI/UCB/max-variance acquisition) | **Partial** | n=30 vs 500 is scale. Variance used to *avoid*, not *explore*, is conceptual. (Docs call this "EGO"; EGO uses expected improvement, which is not in the code.) |
| G2 Geometry audit/repair | BVH, k-d tree, BFS normals, non-manifold repair | Watertight by construction (OpenCASCADE loft + fuse). `isValid()` gated; trimesh watertight **logged, not gated**; `checkMesh` **not gated** | **Replaced** | Legit. No repair is needed for parametric CAD. Missing: mesh-quality gating. |
| G3 SVD/POD of fields | Field snapshots → ~25 modes (99.9 % energy) | Each case reduced to **2 scalars** by in-solver integration (torque, flux). `purgeWrite 2` keeps no snapshot history | **Absent** | Largely scale. 30 snapshots can't support POD in a useful way. Conceptual loss: no flow-structure information and no field-level validation. |
| G4 PCA/Kernel-PCA of geometry | Learn a latent shape basis from raw geometry | The 7 hand-chosen design variables **are** the low-dimensional parameterisation. No PCA, no sensitivity or active-subspace analysis | **Replaced** | Legit for a parametric model. Missing: any test of whether the 7 variables are all identifiable (GP length-scales exist but are never inspected). |
| G5 "Neural operator" | latent shape → POD coefficients (function-space map) | design vector (7) → (C_p, C_q): scalar-output regression | **Different** | Different class. Parametric regression, not operator learning. |
| G6 Transformer, Adam, backprop | Multi-head attention over surface patches; Adam; PyTorch | MLP 7→24→24→2, 842 params; Adam ✔; backprop/autograd ✔; PyTorch ✔; no attention | **Partial** | Legit. There are no spatial tokens to attend over, and 842 params already exceed 60 targets by 14×. |
| G7 PINN / FNO with NS loss | PDE residual + continuity penalty in the loss | **No PDE or continuity term anywhere.** "Physics" = constant non-dimensionalisation + soft monotonicity prior. Physics lives only in the OpenFOAM data generation | **Absent** (label only) | **Conceptual.** The name "PINN" in this repo does not match the GT document's definition. See §7. |
| G8 BNN + VI + NLL + aleatoric/epistemic | Posterior over weights, NLL, two uncertainty types, uncertainty-triggered CFD with fold-back | Deep ensemble (σ across 5–6 members) as an epistemic proxy ✔; MSE not NLL; no aleatoric head; no calibration; **closed loop propose→CFD→record→retrain ✔** (but CFD is triggered at the optimum, not by an uncertainty threshold) | **Partial** | Deep ensembles are a legitimate BNN substitute. The gap is that the ensemble spread is **not calibrated** (§6.3). |
| G9 MDO + adjoint + SQP + MDF | Exact gradients through the surrogate, SQP, multi-discipline | `differential_evolution` (gradient-free) with an L-BFGS-B finite-difference polish; single discipline (aero only); constraint by exterior penalty | **Different** | Conceptual. The surrogate is a PyTorch net, so **autograd gradients are free and unused.** No structural or manufacturing discipline, so "MDO" does not apply. |
| G10 MPI, CUDA, DDP, NCCL | Custom kernels, multi-node MPI, DDP | MPI domain decomposition **inside OpenFOAM** (`decomposePar` scotch, `mpirun -np 8`) ✔; one GPU for an 842-parameter net; no custom CUDA, DDP or NCCL; CFD cases run **serially** | **Partial** | Scale. Nothing in this model benefits from DDP. |

**Scoreboard:** 10 conceptual rows → 0 full matches, 4 partial (G1, G6, G8, G10), 2 replaced by design (G2, G4), 2 absent (G3, G7), 2 different (G5, G9).

---

## 5. Where your pipeline goes beyond the GT document

The GT document has none of these. Several are what the GT document's own steps would need to be credible:

1. **Data-quality gating.** Physical-consistency and convergence gates (`dataset_builder.py:53-62`); strict COV thresholds; a separate residual-based audit.
2. **Honest out-of-sample check.** Five surrogate-proposed designs were run through real CFD and the errors reported (+2 % to +51 % relative error), not just LOO scores.
3. **Objective repair.** The airflow/power ratio was identified as unstable and replaced by flow-at-power-budget.
4. **Trust-region penalty** against extrapolation.
5. **Solver debugging recorded.** `MRFnoSlip` vs `noSlip` (blade loading ×4000), the measurement-disk fix (full-plane COV 42 % → disk), serial meshing.
6. **A CFD-confirmed Pareto front** instead of a single optimum.
7. **A physical sanity check passes** (my check, §6.5): the actuator-disk figure of merit is below 1 for all 36 cases.

---

## 6. Verified discrepancies (docs vs. code vs. data)

### 6.1 `optimize()` is undefined, so the documented drivers cannot run
`module5_optimize/optimize.py` defines `Surrogate`, `max_flow_at_power`, `min_power_at_flow`, `pareto` and `main`, but **no `optimize`**. `run_optimization.py:36` and `run_full_pipeline.py:104` do `from optimize import Surrogate, optimize` and fail at import. `optimize.py:149` and `:161` raise `NameError`. The real loop is `al_iter.py` + `al_orchestrator.sh`, which README §9 mentions only as "...". `README §4` still presents `optimize.py` as "maximizing air delivery at a power budget" (true for `max_flow_at_power` only).

### 6.2 Three different "converged/trustworthy" definitions
| Where | Rule | Cases passing |
|---|---|---|
| `parse_results.py:126` `converged` | torque COV < 0.15 and flow COV < 0.10 | 33/36 |
| `dataset_builder.py:60` `usable` (drives training) | both COV < 0.05 (+ physical) | **30/36** |
| `audit_convergence.py:64` `trustworthy` (evidence file) | both COV < 0.05 **and** Ux, p residual < 5e-3 | **24/32** |

`results/evidence/convergence_audit.csv` is stale: 32 rows (the dataset has 36), and it still shows `opt_00` as unconverged (flow COV 0.35, 0.0074 W, 0.33 m³/min) while the dataset has the re-solved `opt_00` as converged (0.019 W, 2.62 m³/min). `FINAL_REPORT.md` quotes both "24/32" and "30/36". `doe2_001` fails the residual audit (Ux 0.0113) but is `usable` and enters training.

### 6.3 The ensemble uncertainty is not calibrated
README and `surrogate_pinn.py` describe the ensemble spread as calibrated / honest. On the only three candidates that stored a σ:

| case | predicted flow | CFD flow | error | σ (ensemble) | \|error\| / σ |
|---|---|---|---|---|---|
| opt_00 | 3.60 | 2.62 | +0.97 | 0.08 | **12.3** |
| opt_01 | 3.44 | 2.87 | +0.58 | 0.10 | **6.1** |
| opt_02 | 2.24 | 2.19 | +0.05 | 0.03 | 1.5 |

All **five** candidates (the 3 above plus `sweep_15` +1.62 and `sweep_20` +3.37) were over-predicted. The 2 % agreement on `opt_02` is one lucky point; the median relative flow error of the five is 39 %. This is n=3 (n=5 for the sign), so treat it as strong evidence of miscalibration direction, not a calibrated estimate. The `opt_00` proposal sat at `dist_to_data = 0.59997`, exactly on the 0.6 trust-region boundary, so that penalty was binding.

### 6.4 The 1 M sweep picks are biased by construction
`surrogate_sweep.py:103-110` takes `argmax(predicted flow)` over 1 M points using the ensemble mean only. The maximum of a million noisy predictions selects the largest positive error (winner's curse), and the sweep has no σ or trust term. Both sweep picks over-predicted by 37 % and 51 %. The "1.6 s / 615 k designs/s" throughput figure is cited in README, but `results/evidence/surrogate_sweep.json`, listed in the README evidence table, **is not in the repo**.

### 6.5 Physical sanity check (passes)
Actuator-disk figure of merit FoM = ρQ³/(2A²)/P with A = π·R_tip², computed over all 36 cases: min 0.0003, median 0.004, **max 0.74, none above 1**. Optimised `opt_02` is 0.45; `doe_001` 0.66; `sweep_20` 0.74. The baseline is 0.001, i.e. essentially not pumping. A flat cambered plate at chord Reynolds number ≈ 1e4 reaching FoM 0.66–0.74 is on the optimistic edge. This is not evidence of error, but it justifies a mesh-independence test and a sensitivity test on the measurement disk (r=0.16 m is larger than R_tip=0.148 m, so it may include entrained flow). Neither test exists in the repo.

### 6.6 The twist bound is active, so the optimum lies outside the design box
`twist_deg` is within 3 % of its lower bound (−18°) in **6 of the 7** best or proposed designs I checked (`opt_00–02`, `sweep_15`, `sweep_20`, and `doe_011`; only `doe2_007` is clear of it). Other active bounds: `camber` at its lower bound in `opt_00` and `opt_01`; `thickness_tc` at its lower bound in `opt_01` and `opt_02`; `tip_taper` at its lower bound in `opt_01` and `sweep_15`. README's "strong blade washout ≈ −18°" is partly an artefact of where the box was drawn.

### 6.7 The DOE union is not an LHS
Batch 1 was generated for 16 points but only 12 were run (strata occupied per dimension: 9–11 of 12). Batch 2 (seed 2, 18 points) is a perfect LHS (18/18 per dimension). The union of 30 occupies **19–24 of 30** strata per dimension, so the stratification guarantee quoted from the GT document does not hold for the real training set.

### 6.8 Validation lag and mismatch
- LOO is on 25 points; the training set is now 30.
- LOO trains a **single** MLP at 2500 epochs; production uses a 5–6 member ensemble at 2000–3000 epochs, so the ensemble and its uncertainty have never been validated directly.
- The training-curve plot shows training loss only.
- `results/surrogate_report.json` (10 points, 88 % / 122 % / 84 % LOO MAPE) is a leftover from `surrogate.py`.
- GP wins on flow R² (0.77 vs 0.55); the "winner" is chosen on flow MAPE only.

### 6.9 Reproducibility
Hard-coded machine paths: `/mnt/f/CFD ML CAD Major project/…`, `/home/harsh/fan_cfd`, `C:\Users\Harsh\…\python.exe`, `C:\Program Files\FreeCAD 1.0\…`. The GrabCAD STEP, the OpenFOAM case directories and logs are not in the repo. README says `finalize_all.py` "regenerates everything"; in practice it needs the WSL case directories and does not retrain the surrogate.

---

## 7. Closing the gap: options, ranked by value per effort

The first three give the largest increase in agreement with the GT document's concepts for the lowest effort.

1. **Rename or truly constrain the "PINN" (hours).** Either rename it ("monotonicity-regularised MLP ensemble") or add a real scalar physics term. The cheapest real one is the actuator-disk bound FoM ≤ 1 (i.e. P ≥ ρQ³/2A²) as a hinge penalty; all 36 data points already satisfy it. For the non-dimensionalisation to matter, add rpm as an input (even 3–4 rpm values) so C_p and C_q are actually tested.
2. **Use the gradient (hours).** The surrogate is already a PyTorch net. Replace or follow the DE with multi-start L-BFGS/SLSQP on autograd gradients; this is the honest scalar analogue of G9's "adjoint through the surrogate".
3. **Calibrate or drop the uncertainty claim (hours).** Add a coverage check (is \|error\| < 2σ about 95 % of the time under LOO with the *ensemble*?), then either inflate σ by a fitted factor or switch to a GP σ. If kept as is, stop calling it calibrated.
4. **Make uncertainty explore, not only avoid (hours).** Add an acquisition (EI, or max-σ among high-predicted-flow points) beside the `q − σ` exploitation term, and use it for the sweep picks too.
5. **Widen `twist_deg` below −18° (one CFD batch).** Test whether the optimum moves further out.
6. **Single DOE (days of CFD).** Re-run the 4 missing batch-1 designs, or replace batch 1 with one 30-point LHS, to restore stratification.
7. **Gate `checkMesh`, add a mesh-independence study (days).** Needed before the FoM 0.66–0.74 values can be defended.
8. **Field-level extension (large).** Save fields every case, add POD (G3). This is the first step toward any operator-learning claim.

Items that **should stay as they are** for a 30-sample, 7-parameter problem: no Transformer, no DDP, no Kernel-PCA. They would add parameters faster than data.

---

## 8. How to describe this project accurately next to the GT document

If this comparison is for an application document: the project is best described as **"surrogate-assisted optimisation with a CFD-in-the-loop active-learning cycle (scalar-QoI surrogate, deep-ensemble uncertainty, gradient-free optimiser)"**. It sits in the area of G1, G8 and G9. Describing it as a PINN, neural operator or field-reduction pipeline would be challenged by anyone who has taken the GT courses, because G3, G4, G5 and G7 are absent.
