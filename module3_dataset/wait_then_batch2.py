"""Wait for the first DOE batch to finish (all doe_* cases have results), then
launch the doe2 batch. Lets the CFD data generation run unattended."""
import glob, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
PY = sys.executable

# wait until at least 12 doe_* results exist (batch 1 target ~13 incl doe_000)
def n_done(prefix):
    return len([p for p in glob.glob(os.path.join(RESULTS, "cfd", prefix + "*.json"))
                if not p.endswith(".meshsolve.log")])

print("waiting for batch 1 (doe_*) to reach >=12 results...", flush=True)
stable = 0
last = -1
while True:
    n = n_done("doe_")
    if n != last:
        print("  doe_ results:", n, flush=True)
        last = n
    # consider done when count stops growing for ~2 min and >=12
    if n >= 12:
        stable += 1
    else:
        stable = 0
    if stable >= 8:   # ~8*15s = 2 min stable
        break
    time.sleep(15)

print("batch 1 done (%d results). Launching doe2 batch..." % n_done("doe_"), flush=True)
subprocess.run([PY, os.path.join(HERE, "run_doe.py"),
                "--doe", os.path.join(RESULTS, "doe2.csv"),
                "--n", "18", "--endtime", "800", "--nproc", "8"], check=False)
print("doe2 batch finished. Total results:", n_done("doe"), flush=True)
