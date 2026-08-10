#!/usr/bin/env bash
# Build and run the original Fortran 3dmhd as a golden reference on a tiny
# grid, dumping full state snapshots (gstate.<tag>.<rank>.<iter>) that the
# C++ tests compare against.
#
# Config (edit here or via env):
#   GOLD_NPX GOLD_NPY GOLD_NPZ  global grid sizes
#   GOLD_NPEY GOLD_NPEZ         processor grid (must divide NPY/NPZ)
#   GOLD_NTOTAL GOLD_NSTEP0     run length / dump interval
set -Eeuo pipefail

cd "$(dirname "$0")"
GOLD_DIR="$(pwd)"
SRC="$GOLD_DIR/fortran"
BUILD="$GOLD_DIR/build"
OUT="$GOLD_DIR/ref"

NPX="${GOLD_NPX:-16}"; NPY="${GOLD_NPY:-16}"; NPZ="${GOLD_NPZ:-32}"
NPEY="${GOLD_NPEY:-2}"; NPEZ="${GOLD_NPEZ:-2}"
NTOTAL="${GOLD_NTOTAL:-2}"; NSTEP0="${GOLD_NSTEP0:-1}"
NP="${GOLD_NP:-4}"

export PATH="/scratch/zyi/.kez/env/mpis/openmpi-5.0.10-system/openmpi/bin:$PATH"

rm -rf "$BUILD" "$OUT"
mkdir -p "$BUILD" "$OUT"

# 1) Assemble patched sources: tiny topology + run-length override.
cp "$SRC"/3dmhd.f    "$BUILD"/3dmhd.f
cp "$SRC"/3dmhdsub.f "$BUILD"/3dmhdsub.f
cp "$SRC"/3dmhdset.f "$BUILD"/3dmhdset.f
cp "$SRC"/gstate_sub.f "$BUILD"/gstate_sub.f

# Topology + grid sizes in the parameter file.
sed -E \
  -e "s/PARAMETER\(NPEY=[0-9]+,NPEZ=[0-9]+,NPE=NPEY\*NPEZ\)/PARAMETER(NPEY=${NPEY},NPEZ=${NPEZ},NPE=NPEY*NPEZ)/" \
  -e "s/PARAMETER\(NPX=[0-9]+,NPY=[0-9]+,NPZ=[0-9]+\)/PARAMETER(NPX=${NPX},NPY=${NPY},NPZ=${NPZ})/" \
  "$SRC"/3dmhdparam.f > "$BUILD"/3dmhdparam.f

# Run length: NTOTAL and NSTEP0 in the setup routine.
sed -E \
  -e "s/IPAR\(03\)  = [0-9]+/IPAR(03)  = ${NTOTAL}/" \
  -e "s/IPAR\(04\)  = [0-9]+/IPAR(04)  = ${NSTEP0}/" \
  "$BUILD"/3dmhdset.f > "$BUILD"/3dmhdset.f.tmp && mv "$BUILD"/3dmhdset.f.tmp "$BUILD"/3dmhdset.f

# Also force a fixed (non-hardcoded) orientation comment; keep the original
# MYPEY=MOD(MYPE,NPEY) / MYPEZ=MYPE/NPEY mapping.

# 2) Compile.
(
  cd "$BUILD"
  mpifort -g -O2 -std=legacy -ffixed-form -c 3dmhdsub.f
  mpifort -g -O2 -std=legacy -ffixed-form -c 3dmhdset.f
  mpifort -g -O2 -std=legacy -ffixed-form -c gstate_sub.f
  mpifort -g -O2 -std=legacy -ffixed-form -o 3dmhd.exe 3dmhd.f \
      3dmhdsub.o 3dmhdset.o gstate_sub.o
)
echo "Built golden reference: $BUILD/3dmhd.exe"

# 3) Run.
cd "$OUT"
echo "Running: mpirun -np ${NP} $BUILD/3dmhd.exe"
mpirun -np "$NP" --oversubscribe "$BUILD/3dmhd.exe" > run.log 2>&1
echo "Exit: $?"
echo "--- run.log tail ---"
tail -20 run.log
echo "--- gstate files ---"
ls -1 gstate.* 2>/dev/null | head -20
echo "count: $(ls -1 gstate.* 2>/dev/null | wc -l)"
