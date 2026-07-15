#!/bin/bash
# Bootstrap: copy the Windows-side template + a per-design STL into a
# space-free Linux path, then set up + mesh (and optionally solve).
# Args: NAME OMEGA ENDTIME WRITEINT NPROC [mesh|run]
# Expects the metres-STL at:  <module2>/staging/blade_<NAME>.stl
set -e
NAME=${1:-baseline}
OMEGA=${2:-36.65}
ENDTIME=${3:-2000}
WRITEINT=${4:-200}
NPROC=${5:-1}
MODE=${6:-mesh}

SRC="/mnt/f/CFD ML CAD Major project/fan_pipeline/module2_openfoam"
ROOT="/home/harsh/fan_cfd"
TPL="$ROOT/_template"
STL="$ROOT/_stl/blade_${NAME}.stl"

mkdir -p "$ROOT/_stl"
rm -rf "$TPL"; mkdir -p "$TPL"
cp -r "$SRC/case_template/." "$TPL/"

# per-design STL (fallback to generic staging/blade.stl for baseline test)
if [ -f "$SRC/staging/blade_${NAME}.stl" ]; then
    cp "$SRC/staging/blade_${NAME}.stl" "$STL"
else
    cp "$SRC/staging/blade.stl" "$STL"
fi

bash "$SRC/scripts/setup_and_mesh.sh" "$ROOT/$NAME" "$TPL" "$STL" "$OMEGA" "$ENDTIME" "$WRITEINT" "$NPROC"

if [ "$MODE" = "run" ]; then
    bash "$SRC/scripts/solve.sh" "$ROOT/$NAME" "$NPROC"
fi
