#!/bin/bash
# Detached WSL orchestrator for the active-learning optimization loop.
# Kill-proof: runs via setsid; calls Windows python.exe (interop, Windows-path
# args) for the quick ML/CAD/parse steps, and OpenFOAM natively for the CFD.
# Args: N_ITERS ENDTIME NPROC
source /opt/openfoam12/etc/bashrc 2>/dev/null || true
NITERS=${1:-8}
ENDTIME=${2:-800}
NPROC=${3:-8}

WINPY="/mnt/c/Users/Harsh/AppData/Local/Programs/Python/Python312/python.exe"
ALPY='F:\CFD ML CAD Major project\fan_pipeline\al_iter.py'
SRC="/mnt/f/CFD ML CAD Major project/fan_pipeline/module2_openfoam"
PROG="/home/harsh/fan_cfd/active_learning.progress"

echo "AL start $(date)" > "$PROG"

# Phase A: wait for doe2 batch to finish
while ! grep -q "batch complete" /home/harsh/fan_cfd/batch_doe2.progress 2>/dev/null; do
    echo "waiting for doe2 ($(ls -d /home/harsh/fan_cfd/doe2_*/postProcessing 2>/dev/null | wc -l) solved) $(date)" >> "$PROG"
    sleep 60
done
echo "doe2 complete $(date)" >> "$PROG"

# Phase B: parse any straggler doe2 cases
"$WINPY" "$ALPY" parse_doe2 >> "$PROG" 2>&1

# Phase D: active-learning loop
for it in $(seq 0 $((NITERS-1))); do
    name=$(printf "opt_%02d" "$it")
    if [ -f "/home/harsh/fan_cfd/$name/postProcessing/rotorForces/0/forces.dat" ] \
       && [ -f "/mnt/f/CFD ML CAD Major project/fan_pipeline/results/cfd/$name.json" ]; then
        echo ">>> $name already done, skip" >> "$PROG"; continue
    fi
    echo ">>> $name PROPOSE $(date)" >> "$PROG"
    "$WINPY" "$ALPY" propose --name "$name" >> "$PROG" 2>&1 || { echo "$name propose FAILED" >> "$PROG"; break; }
    echo ">>> $name SOLVE $(date)" >> "$PROG"
    bash "$SRC/scripts/stage_and_run.sh" "$name" 36.6519 "$ENDTIME" 100 "$NPROC" run \
        >> "/home/harsh/fan_cfd/log_${name}.txt" 2>&1
    echo ">>> $name RECORD $(date)" >> "$PROG"
    "$WINPY" "$ALPY" record --name "$name" >> "$PROG" 2>&1
done

# Phase F: finalize (best design CAD + screenshots + report)
echo ">>> FINALIZE $(date)" >> "$PROG"
"$WINPY" "$ALPY" finalize >> "$PROG" 2>&1
echo "AL complete $(date)" >> "$PROG"
