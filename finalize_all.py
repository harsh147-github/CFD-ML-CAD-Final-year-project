"""
One-shot final assembly of the whole pipeline's results + evidence. Run after the
active-learning loop finishes. Regenerates everything from the current CFD data so
the deliverable is fully reproducible:

  1. dataset_builder     -> dataset.csv/.parquet (all cases, strict trust gate)
  2. audit_convergence   -> evidence/convergence_audit.csv
  3. validate_surrogates -> evidence/surrogate_parity.png + validation.json (PINN vs GP LOO)
  4. make_pareto         -> evidence/pareto_airflow_power.png + pareto_front.csv
  5. make_residual_plots -> evidence/residuals_<case>.png (baseline + best designs)
  6. al_iter finalize    -> best_design.json + optimized_fan.step/.stl + CFD screenshots
  7. generate_report     -> docs/FINAL_REPORT.md

Every number in the report traces to a real OpenFOAM solve.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
RES = os.path.join(ROOT, "results")


def run(label, args):
    print("\n=== %s ===" % label, flush=True)
    r = subprocess.run([PY] + args, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    # print only the informative lines
    for line in out.splitlines():
        if any(k in line for k in ("MAPE", "Pareto", "Wrote", "OPTIMIZED", "trustworthy",
                                   "Dataset:", "winner", "Trustworthy", "FINALIZED",
                                   "power=", "flow=", "Error", "Traceback", "Saved")):
            print("  " + line, flush=True)
    if r.returncode != 0:
        print("  [nonzero exit %d]" % r.returncode, flush=True)


def main():
    # record any CFD-confirmed surrogate-sweep candidates first
    for nm in ("sweep_15", "sweep_20"):
        case = "//wsl.localhost/Ubuntu/home/harsh/fan_cfd/%s/postProcessing/rotorForces" % nm
        if os.path.isdir(case) and not os.path.exists(os.path.join(RES, "cfd", nm + ".json")):
            run("0. record %s" % nm, [os.path.join(ROOT, "al_iter.py"), "record", "--name", nm])
    run("1. dataset", [os.path.join(ROOT, "module3_dataset", "dataset_builder.py"),
                       "--in", os.path.join(RES, "cfd"), "--out", os.path.join(RES, "dataset")])
    run("2. convergence audit", [os.path.join(ROOT, "module3_dataset", "audit_convergence.py")])
    run("3. surrogate validation (PINN vs GP LOO)",
        [os.path.join(ROOT, "module4_pinn", "validate_surrogates.py")])
    run("4. Pareto front", [os.path.join(ROOT, "module5_optimize", "make_pareto.py")])
    run("4b. surrogate 1e6-design sweep",
        [os.path.join(ROOT, "module5_optimize", "surrogate_sweep.py"), "--n", "1000000"])
    run("5. residual convergence plots",
        [os.path.join(ROOT, "module3_dataset", "make_residual_plots.py"),
         "baseline", "doe_011", "doe_001", "doe2_007", "opt_00", "sweep_15", "sweep_20"])
    run("6. finalize best design + CAD",
        [os.path.join(ROOT, "al_iter.py"), "finalize"])
    run("7. report", [os.path.join(ROOT, "module6_validation", "generate_report.py"),
                      "--out", os.path.join(ROOT, "docs", "FINAL_REPORT.md")])
    print("\n=== PIPELINE FINALIZED ===", flush=True)


if __name__ == "__main__":
    main()
