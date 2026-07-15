#!/bin/bash
# Set up an OpenFOAM case from the template and mesh it (blockMesh + snappy).
# Args: CASE_DIR TEMPLATE_DIR STL_PATH OMEGA ENDTIME WRITEINT NPROC
# Source OF env BEFORE set -e (its bashrc returns nonzero under set -e).
source /opt/openfoam12/etc/bashrc 2>/dev/null || true
set -e

CASE=$1; TEMPLATE=$2; STL=$3; OMEGA=$4; ENDTIME=$5; WRITEINT=$6; NPROC=$7

echo ">>> Setting up case: $CASE"
rm -rf "$CASE"
mkdir -p "$CASE"
cp -r "$TEMPLATE"/0.orig "$CASE"/0.orig
cp -r "$TEMPLATE"/constant "$CASE"/constant
cp -r "$TEMPLATE"/system   "$CASE"/system
mkdir -p "$CASE"/constant/geometry
cp "$STL" "$CASE"/constant/geometry/blade.stl

# token substitution
sed -i "s/__OMEGA__/$OMEGA/g"       "$CASE"/constant/MRFProperties
sed -i "s/__ENDTIME__/$ENDTIME/g"   "$CASE"/system/controlDict
sed -i "s/__WRITEINT__/$WRITEINT/g" "$CASE"/system/controlDict
sed -i "s/__NPROC__/$NPROC/g"       "$CASE"/system/decomposeParDict

cd "$CASE"
echo ">>> blockMesh"
blockMesh > log.blockMesh 2>&1
echo ">>> surfaceFeatures"
surfaceFeatures > log.surfaceFeatures 2>&1
# Mesh serially: it is small (~150k cells) and snaps in <1 min; this avoids
# OF12 parallel-mesh reconstruction issues. The SOLVE runs in parallel.
echo ">>> snappyHexMesh (serial)"
snappyHexMesh -overwrite > log.snappyHexMesh 2>&1
echo ">>> checkMesh"
checkMesh -constant > log.checkMesh 2>&1 || true

echo ">>> DONE MESH. Summary:"
echo "--- cells ---"
grep -E "cells:" log.checkMesh | head -1
echo "--- rotor cellZone ---"
grep -iE "rotor" log.checkMesh || echo "(no rotor line in checkMesh)"
echo "--- mesh OK? ---"
grep -E "Mesh OK|Failed" log.checkMesh | tail -2
echo "--- patches ---"
grep -A20 "Boundary definition" log.checkMesh 2>/dev/null | head -5 || true
