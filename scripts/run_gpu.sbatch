#!/bin/bash
#===============================================================================
# run_gpu.sbatch - rast-3dmhd GPU backend (up to 8 ranks / GPUs on one node).
#
#   Build first:  cd <workspace>/Rast-3dmhd && make BACKEND=gpu
#   Submit:       sbatch --gres=gpu:8 scripts/run_gpu.sbatch
#
# One MPI rank per GPU.  Each rank is pinned to its own device via
# CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK (rank i <-> device i on the
# node), so the app's --gpu 0 (cudaSetDevice) hits the granted GPU.
#
# Grid/process grid: npey | 8 (ranks) and npy, npez = 8/npey | npz, npez >= 2.
# For 504x504x2048: npey=4 -> npez=2.
#
# Edit the values below directly (this script does not take environment
# overrides).  Everything resolves standalone via RPATHs - no PATH or
# LD_LIBRARY_PATH manipulation is needed.
#===============================================================================
#SBATCH --job-name=rast3dmhd-gpu
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=8                # one MPI rank per GPU
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --open-mode=append

set -euo pipefail

# --- run configuration --------------------------------------------------------
NP=8                                     # MPI ranks = GPUs
NPY=504                                  # global grid points in y
NPZ=2048                                 # global grid points in z
NPEY=4                                   # processors in y (npez = NP/NPEY)
NTOTAL=1000                              # RK steps
NSTEP0=100                               # dump/log every NSTEP0 steps
WORKDIR=/scratch/zyi/Applications/rast3dmhd-runs
BIN=/scratch/zyi/.kez/env/applications/test-3dmhd/rast-3dmhd/bin/3dmhd
MPIRUN=/scratch/zyi/.kez/env/mpis/openmpi-5.0.10-system/openmpi/bin/mpirun

# --- validation ---------------------------------------------------------------
NPEZ=$((NP / NPEY))
[ -x "$BIN" ]    || { echo "error: binary not executable: $BIN (build with: make BACKEND=gpu)"; exit 1; }
# A CPU-only build would abort mid-job on --backend gpu; fail fast instead.
# (nm reports the C++ mangled name "...step_gpu...", so match the bare symbol.)
nm "$BIN" 2>/dev/null | grep -q 'step_gpu' \
    || { echo "error: $BIN is not a GPU build (rebuild with: make BACKEND=gpu)"; exit 1; }
[ $((NP % NPEY)) -eq 0 ]   || { echo "error: npey $NPEY does not divide NP $NP"; exit 1; }
[ $((NPY % NPEY)) -eq 0 ]  || { echo "error: npy $NPY not divisible by npey $NPEY"; exit 1; }
[ "$NPEZ" -ge 2 ]          || { echo "error: npez $NPEZ must be >= 2"; exit 1; }
[ $((NPZ % NPEZ)) -eq 0 ]  || { echo "error: npz $NPZ not divisible by npez $NPEZ"; exit 1; }
[ "$NP" -le 8 ]            || { echo "error: this script pins GPUs to 0..7 per node; NP=$NP > 8"; exit 1; }

mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "== rast-3dmhd GPU job: gpus=$NP grid 504x${NPY}x${NPZ} npey=$NPEY npez=$NPEZ steps=$NTOTAL =="
T0=$(date +%s)

# Per-rank GPU pinning; the benign libnuma dlopen message is filtered.
"$MPIRUN" --allow-run-as-root \
          -np "$NP" \
          bash -c 'export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK; exec "$@"' _ \
          "$BIN" --backend gpu --npy "$NPY" --npz "$NPZ" --npey "$NPEY" \
                 --ntotal "$NTOTAL" --nstep0 "$NSTEP0" \
          2>&1 | grep -a -v -E 'LOG_CAT_SBGP|libnuma\.so'

T1=$(date +%s)
echo "== rast-3dmhd GPU job finished: elapsed=$((T1 - T0))s =="
