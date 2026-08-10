#!/bin/bash
#===============================================================================
# run_gpu.sbatch - run the GPU backend of rast-3dmhd (up to 8 GPUs, one node).
#
#   Build first (choose the GPU backend!):
#       cd <workspace>/Rast-3dmhd && make clean && make BACKEND=gpu -j
#   Then submit:
#       sbatch --gres=gpu:8 scripts/run_gpu.sbatch
#       sbatch --gres=gpu:4 --ntasks=4 --export=ALL,NPEY=2 scripts/run_gpu.sbatch
#
# One MPI rank per GPU.  Each rank is pinned to its own device via
# CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK (0..NGPUS-1 within the node),
# so the app's --gpu 0 (cudaSetDevice) resolves to the physical device Slurm
# granted to that rank.  The full state stays GPU-resident between steps; only
# the per-step dt reduction and the periodic dumps cross to the host.
#
# Constraints (validated below): npey divides NPY; npez = NP/npey divides NPZ
# and is >= 2.  8 ranks, default grid 504x504x2048: npey=4 -> npez=2 (or
# npey=2 -> npez=4).
#===============================================================================
#SBATCH --job-name=rast3dmhd-gpu
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=8               # one MPI rank per GPU
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --output=logs/%x_%j.out       # job log (also see LOGDIR below)
#SBATCH --error=logs/%x_%j.err
#SBATCH --open-mode=append

set -euo pipefail

#--- configurable knobs --------------------------------------------------------
NP=${NP:-$SLURM_NTASKS}        # MPI ranks = GPUs used
NPY=${NPY:-504}                # global grid points in y
NPZ=${NPZ:-2048}               # global grid points in z
NPEY=${NPEY:-4}                # processors in y (npez = NP/NPEY derived)
NTOTAL=${NTOTAL:-1000}         # RK steps to run
NSTEP0=${NSTEP0:-100}          # dump/log every NSTEP0 steps
WORKDIR=${WORKDIR:-$HOME/rast3dmhd-runs}   # where the gstate.* dumps land

# Paths to the Kez environment the binary was built/linked against.
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
[ -x "$BIN" ] || { echo "error: binary not found/executable: $BIN (build with: make BACKEND=gpu)"; exit 1; }
[ $((NP % NPEY)) -eq 0 ] || { echo "error: npey $NPEY does not divide NP $NP"; exit 1; }
[ $((NPY % NPEY)) -eq 0 ] || { echo "error: npy $NPY not divisible by npey $NPEY"; exit 1; }
[ "$NPEZ" -ge 2 ]      || { echo "error: npez $NPEZ must be >= 2"; exit 1; }
[ $((NPZ % NPEZ)) -eq 0 ] || { echo "error: npz $NPZ not divisible by npez $NPEZ"; exit 1; }
[ "$NP" -le 8 ] || { echo "error: this script pins GPUs to 0..7 per node; NP=$NP > 8"; exit 1; }

mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "== rast-3dmhd GPU job =="
echo "   gpus=$NP (npey=$NPEY npez=$NPEZ)  grid 504x${NPY}x${NPZ}  steps=$NTOTAL (dump every $NSTEP0)"
echo "   binary=$BIN  workdir=$WORKDIR"
echo "   log: logs/rast3dmhd-gpu_${SLURM_JOB_ID:-?}.out   (NIT + timing lines land here)"
echo "   submitted=$(date '+%Y-%m-%d %H:%M:%S')"
T0=$(date +%s)

# Per-rank GPU pinning: each MPI rank sees only the physical GPU Slurm granted
# it (rank i <-> device i on a single shared node).
# OpenMPI's binding detection dlopens libnuma (falls back gracefully); the
# message is harmless. Merge mpirun's streams through a filter so real
# messages and the job banner land together in the .out log while the noise
# is dropped. (No UCX/CUDA filter: on a GPU job a cuDeviceGetCount failure
# would be a real error to keep visible.)
mpirun --allow-run-as-root \
       --map-by ppr:${NP}:node \
       -np "$NP" \
       bash -c 'export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK; exec "$@"' _ \
       "$BIN" --backend gpu --npy "$NPY" --npz "$NPZ" --npey "$NPEY" \
              --ntotal "$NTOTAL" --nstep0 "$NSTEP0" \
       2>&1 | grep -a -v -E 'LOG_CAT_SBGP|libnuma\.so'

T1=$(date +%s)
echo "== rast-3dmhd GPU job finished (exit $?) =="
echo "   finished=$(date '+%Y-%m-%d %H:%M:%S')  elapsed=$((T1 - T0))s  ~$(awk -v e=$((T1 - T0)) -v s="$NTOTAL" 'BEGIN{ printf "%.2f s/step", e/s }')"
