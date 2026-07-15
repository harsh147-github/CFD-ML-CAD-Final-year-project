# Evidence archive — proofs, not claims

Every headline number in this project traces to a real OpenFOAM solve or a
validation artifact stored here. Nothing is asserted that a file below does not support.

## Data integrity
- **`convergence_audit.csv`** — per-case audit of every solved CFD case: final
  Ux/p residuals, torque coefficient-of-variation, flow coefficient-of-variation,
  and a `trustworthy` flag (both torque AND flow stable, cov < 0.05). This is how
  8/32 lightly-loaded, unsteady-wake cases were caught and excluded from training.

## Surrogate validation (leave-one-out — tests generalisation to unseen geometry)
- **`surrogate_validation.json`** — MAPE + R² for BOTH the PINN and the GP on the
  25 trustworthy points. PINN wins (power R² 0.90 vs 0.79; Service-Value R² 0.73 vs −1.98).
- **`surrogate_parity.png`** — LOO predicted-vs-actual scatter, power & airflow, both
  models. Shows the PINN is tight on power and on the data-dense airflow region, and
  under-predicts the sparse high-airflow designs (why surrogate optima are CFD-confirmed).
- **`pinn_training_curve.png`** — PINN training-loss decay (sanity that it trains).

## CFD convergence proofs
- **`residuals_<case>.png`** — for baseline and the optimized/best designs: initial
  residuals (Ux,Uy,Uz,p) vs iteration + the airflow-through-disk settling to a flat
  value. Demonstrates the reported air-delivery numbers are converged, not transient.

## How to regenerate
```
python module3_dataset/audit_convergence.py        # -> convergence_audit.csv
python module4_pinn/validate_surrogates.py         # -> parity + validation json + training curve
python module3_dataset/make_residual_plots.py baseline doe_011 doe_001   # -> residual plots
```

## Result provenance
- Full CFD dataset (design vars + operating condition + CFD outputs + cov flags):
  `../dataset.csv` / `../dataset.parquet`.
- Per-case raw JSON (torque, power, flow, covs, convergence): `../cfd/*.json`.
- Baseline-vs-optimized comparison: `../best_design.json`.
- Optimized geometry: `../optimized_fan.step` / `.stl`.
- Narrative report: `../../docs/FINAL_REPORT.md`; build history: `../../docs/BUILD_LOG.md`.
