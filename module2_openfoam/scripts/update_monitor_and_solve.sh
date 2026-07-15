#!/bin/bash
# Refresh the flow-measurement disk + controlDict into an existing meshed case,
# set endTime, and re-solve (no re-meshing). Args: NAME ENDTIME NPROC
source /opt/openfoam12/etc/bashrc 2>/dev/null || true
set -e
NAME=${1:-baseline}
ENDTIME=${2:-1000}
NPROC=${3:-8}

SRC="/mnt/f/CFD ML CAD Major project/fan_pipeline/module2_openfoam/case_template"
C="/home/harsh/fan_cfd/$NAME"

mkdir -p "$C/constant/geometry"
cp "$SRC/constant/geometry/measure_disk.stl" "$C/constant/geometry/"
cp "$SRC/system/controlDict" "$C/system/controlDict"
# use sed (this OF12 build ships a flaky debug foamDictionary that core-dumps)
sed -i "s/__ENDTIME__/$ENDTIME/g; s/__WRITEINT__/100/g" "$C/system/controlDict"
echo "disk: $(ls "$C/constant/triSurface/")  endTime=$ENDTIME"

bash "$SRC/../scripts/solve.sh" "$C" "$NPROC"
