#!/bin/bash
#===============================================================================
# run_cpu.sbatch - run the CPU backend of rast-3dmhd (up to 128 cores, one node).
#
#   Build first (choose the CPU backend!):
#       cd <workspace>/Rast-3dmhd && make clean && make -j
#   Then submit (override knobs with --export / --ntasks):
#       sbatch --ntasks=128 scripts/run_cpu.sbatch
#       sbatch --ntasks=64  --export=ALL,NTOTAL=200 scripts/run_cpu.sbatch
#
# Constraints the chosen run must satisfy (the code validates these too):
#   * npey  divides both NP  (MPI ranks) and NPY
#   * npez = NP / npey divides NPZ, and npez >= 2
#   Default 504x504x2048, 128 ranks -> npey=8, npez=16 (both divide exactly).
#   Other 128-rank choices: npey=4 -> npez=32, npey=2 -> npez=64, npey=1 -> npez=128.
#===============================================================================
#SBATCH --job-name=rast3dmhd-cpu
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=128              # MPI ranks = physical cores used (<= 128/node)
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x_%j.out        # job log (also see LOGDIR below)
#SBATCH --error=logs/%x_%j.err
#SBATCH --open-mode=append

set -euo pipefail

#--- configurable knobs --------------------------------------------------------
NP=${NP:-$SLURM_NTASKS}        # MPI ranks (cores): "up to 128"
NPY=${NPY:-504}                # global grid points in y
NPZ=${NPZ:-2048}               # global grid points in z
NPEY=${NPEY:-8}                # processors in y (npez = NP/NPEY derived)
NTOTAL=${NTOTAL:-1000}         # RK steps to run
NSTEP0=${NSTEP0:-100}          # dump/log every NSTEP0 steps
WORKDIR=${WORKDIR:-$HOME/rast3dmhd-runs}   # where the gstate.* dumps land

# Paths to the Kez environment the binary was built/linked against.
# Override if the installation moved (e.g. `make install` into a prefix).
KEZENV=${KEZENV:-/scratch/zyi/.kez/env}
MPIROOT=${MPIROOT:-$KEZENV/mpis/openmpi-5.0.10-system/openmpi}
# Locate the binary.  Under sbatch, $0/BASH_SOURCE point at Slurm's spool copy
# of the script (/var/spool/slurmd/...), so the relative paths there are
# useless - resolve in this order instead:
#   BIN (explicit override) > SLURM_SUBMIT_DIR/bin/3dmhd >
#   $(dirname of this script)/../bin/3dmhd > absolute workspace fallback.
if [ -z "${BIN:+x}" ]; then
    if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -x "$SLURM_SUBMIT_DIR/bin/3dmhd" ]; then
        BIN="$SLURM_SUBMIT_DIR/bin/3dmhd"
    elif [ -x "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)/bin/3dmhd" ]; then
        BIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)/bin/3dmhd"
    else
        BIN=/home/zyi/Kez/workspace/Rast-3dmhd/bin/3dmhd
    fi
fi
#===============================================================================

#--- environment ---------------------------------------------------------------
export PATH="$MPIROOT/bin:$KEZENV/system/bin:$PATH"
# The binary carries an RPATH for the Kez libs; LD_LIBRARY_PATH is belt-and-braces.
export LD_LIBRARY_PATH="$MPIROOT/lib:$MPIROOT/../ucx/lib:$MPIROOT/../pmix/lib:$KEZENV/vendors/hpcx-2.50.rhel.9/hcoll/lib:$KEZENV/system/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

#--- validation ----------------------------------------------------------------
NPEZ=$((NP / NPEY))
[ -x "$BIN" ] || { echo "error: binary not found/executable: $BIN (build with: make)"; exit 1; }
[ $((NP % NPEY)) -eq 0 ] || { echo "error: npey $NPEY does not divide NP $NP"; exit 1; }
[ $((NPY % NPEY)) -eq 0 ] || { echo "error: npy $NPY not divisible by npey $NPEY"; exit 1; }
[ "$NPEZ" -ge 2 ]      || { echo "error: npez $NPEZ must be >= 2"; exit 1; }
[ $((NPZ % NPEZ)) -eq 0 ] || { echo "error: npz $NPZ not divisible by npez $NPEZ"; exit 1; }

mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "== rast-3dmhd CPU job =="
echo "   ranks=$NP (npey=$NPEY npez=$NPEZ)  grid 504x${NPY}x${NPZ}  steps=$NTOTAL (dump every $NSTEP0)"
echo "   binary=$BIN  workdir=$WORKDIR"
echo "   log: logs/rast3dmhd-cpu_${SLURM_JOB_ID:-?}.out   (NIT + timing lines land here)"
echo "   submitted=$(date '+%Y-%m-%d %H:%M:%S')"
T0=$(date +%s)

# On a CPU-only allocation UCX (built with CUDA support) logs a benign
# "cuDeviceGetCount ... initialization error" per rank at MPI init, and
# OpenMPI's binding detection dlopens libnuma (falls back gracefully). Both
# are harmless; merge mpirun's streams through a filter so real messages
# (the driver keeps everything on stderr) and the job banner land together in
# the .out log while the noise is dropped.
mpirun --allow-run-as-root \
       --map-by core --bind-to core \
       -np "$NP" \
       "$BIN" --npy "$NPY" --npz "$NPZ" --npey "$NPEY" \
              --ntotal "$NTOTAL" --nstep0 "$NSTEP0" \
       2>&1 | grep -a -v -E 'LOG_CAT_SBGP|libnuma\.so|gdaki\.c:.*cuDeviceGetCount'

T1=$(date +%s)
echo "== rast-3dmhd CPU job finished (exit $?) =="
echo "   finished=$(date '+%Y-%m-%d %H:%M:%S')  elapsed=$((T1 - T0))s  ~$(awk -v e=$((T1 - T0)) -v s="$NTOTAL" 'BEGIN{ printf "%.2f s/step", e/s }')"
