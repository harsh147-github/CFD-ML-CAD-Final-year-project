#!/bin/bash
# Wait for the active-learning loop to finish, then CFD-confirm the top
# surrogate-screened designs (sweep_15, sweep_20). Detached / kill-proof.
source /opt/openfoam12/etc/bashrc 2>/dev/null || true
SRC="/mnt/f/CFD ML CAD Major project/fan_pipeline/module2_openfoam"
PROG="/home/harsh/fan_cfd/sweep_solve.progress"
echo "sweep-solve start $(date)" > "$PROG"

# wait for AL loop
while ! grep -q "AL complete" /home/harsh/fan_cfd/active_learning.progress 2>/dev/null; do
    sleep 20
done
echo "AL loop done, solving sweep candidates $(date)" >> "$PROG"

for name in sweep_15 sweep_20; do
    if [ -f "/home/harsh/fan_cfd/$name/postProcessing/rotorForces/0/forces.dat" ]; then
        echo "$name already solved" >> "$PROG"; continue
    fi
    echo ">>> $name SOLVE $(date)" >> "$PROG"
    bash "$SRC/scripts/stage_and_run.sh" "$name" 36.6519 1200 100 8 run \
        >> "/home/harsh/fan_cfd/log_${name}.txt" 2>&1
    if [ -f "/home/harsh/fan_cfd/$name/postProcessing/rotorForces/0/forces.dat" ]; then
        echo ">>> $name DONE $(date)" >> "$PROG"
    else
        echo ">>> $name FAILED $(date)" >> "$PROG"
    fi
done
echo "sweep-solve complete $(date)" >> "$PROG"
