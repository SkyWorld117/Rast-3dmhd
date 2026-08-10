#!/bin/bash
#===============================================================================
# run_cpu.sbatch - rast-3dmhd CPU backend (up to 128 ranks / cores on one node).
#
#   Build first:  cd <workspace>/Rast-3dmhd && make
#   Submit:       sbatch scripts/run_cpu.sbatch
#
# The grid and process grid below must satisfy: npey | 128 and npy,
# npez = 128/npey | npz, and npez >= 2.  For 504x504x2048 and 128 ranks that
# is npey=8 -> npez=16.
#
# Edit the values below directly (this script does not take environment
# overrides).  Everything resolves standalone: MPIRUN and BIN are absolute
# paths and both the binary and the OpenMPI tree carry RPATHs, so no PATH or
# LD_LIBRARY_PATH manipulation is needed.
#===============================================================================
#SBATCH --job-name=rast3dmhd-cpu
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=128              # MPI ranks = physical cores used
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --open-mode=append

set -euo pipefail

# --- run configuration --------------------------------------------------------
NP=128                                   # MPI ranks (cores)
NPY=504                                  # global grid points in y
NPZ=2048                                 # global grid points in z
NPEY=8                                   # processors in y (npez = NP/NPEY)
NTOTAL=1000                              # RK steps
NSTEP0=100                               # dump/log every NSTEP0 steps
WORKDIR=/scratch/zyi/Applications/rast3dmhd-runs
BIN=/scratch/zyi/.kez/env/applications/test-3dmhd/rast-3dmhd/bin/3dmhd
MPIRUN=/scratch/zyi/.kez/env/mpis/openmpi-5.0.10-system/openmpi/bin/mpirun

# --- validation ---------------------------------------------------------------
NPEZ=$((NP / NPEY))
[ -x "$BIN" ]    || { echo "error: binary not executable: $BIN (build with: make)"; exit 1; }
[ $((NP % NPEY)) -eq 0 ]   || { echo "error: npey $NPEY does not divide NP $NP"; exit 1; }
[ $((NPY % NPEY)) -eq 0 ]  || { echo "error: npy $NPY not divisible by npey $NPEY"; exit 1; }
[ "$NPEZ" -ge 2 ]          || { echo "error: npez $NPEZ must be >= 2"; exit 1; }
[ $((NPZ % NPEZ)) -eq 0 ]  || { echo "error: npz $NPZ not divisible by npez $NPEZ"; exit 1; }

mkdir -p "$WORKDIR"
cd "$WORKDIR"

# One OpenMP thread per rank CPU (this script pins 1 core per rank; for a
# hybrid run give each rank more cores via --cpus-per-task and it scales up).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "== rast-3dmhd CPU job: ranks=$NP grid 504x${NPY}x${NPZ} npey=$NPEY npez=$NPEZ steps=$NTOTAL =="
T0=$(date +%s)

# UCX (a CUDA-enabled build) logs a benign cuDeviceGetCount error per rank on a
# CPU-only allocation, and OpenMPI dlopens libnuma (falls back); both are
# filtered so real output stays visible.
"$MPIRUN" --allow-run-as-root --map-by core --bind-to core \
          -np "$NP" \
          "$BIN" --npy "$NPY" --npz "$NPZ" --npey "$NPEY" \
                 --ntotal "$NTOTAL" --nstep0 "$NSTEP0" \
          2>&1 | grep -a -v -E 'LOG_CAT_SBGP|libnuma\.so|gdaki\.c:.*cuDeviceGetCount'

T1=$(date +%s)
echo "== rast-3dmhd CPU job finished: elapsed=$((T1 - T0))s =="
