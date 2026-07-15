#!/bin/bash
# Solve a meshed fan case with foamRun (incompressibleFluid, MRF, kOmegaSST).
# Args: CASE_DIR NPROC
source /opt/openfoam12/etc/bashrc 2>/dev/null || true
set -e
CASE=$1; NPROC=${2:-1}
cd "$CASE"

# clean any previous solution (time dirs, processors, postProcessing) so the
# run starts fresh from 0 rather than resuming at a previous endTime.
foamListTimes -rm > /dev/null 2>&1 || true
rm -rf processor* postProcessing

# fresh initial fields on the final (snapped) mesh
rm -rf 0
cp -r 0.orig 0

echo ">>> foamRun ($NPROC proc)"
if [ "$NPROC" -gt 1 ]; then
    rm -rf processor*
    decomposePar -force > log.decomposePar 2>&1
    mpirun -np "$NPROC" foamRun -parallel > log.foamRun 2>&1
    reconstructPar -latestTime > log.reconstructPar 2>&1
else
    foamRun > log.foamRun 2>&1
fi

echo ">>> foamRun DONE"
tail -25 log.foamRun | grep -E "Time =|Solving for|continuity" | tail -12
echo ">>> last residuals:"
grep -E "Solving for (Ux|Uy|Uz|p|omega|k)," log.foamRun | tail -6
