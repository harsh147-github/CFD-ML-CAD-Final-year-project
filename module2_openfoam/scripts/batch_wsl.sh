#!/bin/bash
# Detached WSL mesh+solve loop over a set of pre-built STLs. Survives harness
# background-task limits (run via nohup). Skips cases already solved.
# Args: PREFIX ENDTIME NPROC   (e.g. doe2 800 8)
# Expects metres-STLs at <module2>/staging/blade_<PREFIX>_NNN.stl
source /opt/openfoam12/etc/bashrc 2>/dev/null || true
PREFIX=${1:-doe2}
ENDTIME=${2:-800}
NPROC=${3:-8}

SRC="/mnt/f/CFD ML CAD Major project/fan_pipeline/module2_openfoam"
ROOT="/home/harsh/fan_cfd"
PROG="$ROOT/batch_${PREFIX}.progress"
echo "batch start $(date)" > "$PROG"

for stl in "$SRC"/staging/blade_${PREFIX}_*.stl; do
    [ -e "$stl" ] || continue
    base=$(basename "$stl" .stl)      # blade_doe2_000
    name=${base#blade_}               # doe2_000
    case_dir="$ROOT/$name"
    if [ -f "$case_dir/postProcessing/rotorForces/0/forces.dat" ]; then
        echo "$name already solved, skip" >> "$PROG"
        continue
    fi
    echo ">>> $name start $(date)" >> "$PROG"
    bash "$SRC/scripts/stage_and_run.sh" "$name" 36.6519 "$ENDTIME" 100 "$NPROC" run \
        >> "$ROOT/log_${name}.txt" 2>&1
    if [ -f "$case_dir/postProcessing/rotorForces/0/forces.dat" ]; then
        echo ">>> $name DONE $(date)" >> "$PROG"
    else
        echo ">>> $name FAILED $(date)" >> "$PROG"
    fi
done
echo "batch complete $(date)" >> "$PROG"
