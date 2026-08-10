# Rast-3DMHD

GPU-accelerated 3D magnetohydrodynamics. This is a C++/CUDA rewrite of the
Rast 3DMHD Fortran code: the MPI process grid is **parsed at runtime instead
of hardcoded**, and the compute sweeps run either on the CPU or, in a CUDA
build, on NVIDIA GPUs.

The physics core reproduces the original Fortran **bit-for-bit** on the CPU
and (with the default build flags) on the GPU; see
[Reproducibility](#reproducibility).

---

## Building

The project is MPI-only (NPEZ >= 2 is required at runtime), so every build is
an MPI build. All make variables are overridable on the command line.

```sh
make                    # CPU backend -> bin/3dmhd
make BACKEND=gpu        # CUDA backend (needs nvcc + the MPI wrapper to find MPI flags)
make -j8                # parallel build
make install PREFIX=/path   # installs bin/, lib/, include/ under PREFIX
make clean
```

### Compile-time options (make variables)

| Variable | Values | Default | Notes |
|---|---|---|---|
| `BACKEND` | `cpu` / `gpu` | `cpu` | GPU requires the CUDA toolkit; `--backend gpu` at runtime then selects the GPU loop |
| `CXX` | path | `mpicxx` | MPI C++ compiler; for the GPU build it is also queried for MPI include/link flags (`MPI_CFLAGS`/`MPI_LIBS` override those) |
| `NVCC` | path | `$(CUDA_HOME)/bin/nvcc` | CUDA compiler |
| `NVCC_ARCH` | e.g. `sm_90` | `sm_90` | CUDA compute capability (H100/H200) |
| `CXXFLAGS` | string | `-O2 -g -std=c++17 -Wall -Wextra -ffp-contract=off` | Host flags. An empty value falls back to this default |
| `NVCCFLAGS` | string | `-O2 -g -std=c++17 -fmad=false` | User-tunable device flags; structural flags (`-rdc=true --cudart=static -x cu -arch`) are always added |
| `LDFLAGS` / `LDLIBS` | string | empty / `-lm` | Linker flags / extra libs |
| `DUMPMODE` | `periodic` / `final` / `none` | `periodic` | Compile-time checkpoint policy, see [Output](#output) |
| `WITH_TESTS` | `0` / `1` | `0` | Build the GoogleTest suite (`make WITH_TESTS=1 test...`); requires gtest |
| `GTEST_PREFIX` | path | Kez system cellar | Header/lib location for the tests (derives `GTEST_INC` / `GTEST_LIB`) |
| `OPENMP` | `0` / `1` | `1` | Thread the CPU-backend sweeps with OpenMP (hybrid MPI+OpenMP); thread count from `OMP_NUM_THREADS`/affinity. The sweeps write only their own cell and the reductions are exact min/max, so the golden bit-exact result is preserved at any thread count. Only the CPU backend links `-fopenmp`; the GPU backend compiles the same files as nvcc host code, where the pragmas are compiled out |
| `USE_NCCL` | `0` / `1` | `0` | GPU halo exchange via device-side NCCL (`ncclAllGather` exchange + `ncclAllReduce` for dt) instead of host-staged MPI; needs an NCCL (`NCCL_HOME`, or `NCCL_INC`/`NCCL_LIBS`). GPU-only |
| `NCCL_HOME` | path | empty | NCCL prefix (`-I`/`-L`/rpath derived); an empty value looks NCCL up on the default search path |

### Tests

```sh
make WITH_TESTS=1 test           # single-process (non-MPI) gtest binaries
make WITH_TESTS=1 test_multi     # MPI tests under mpirun -np 4
make BACKEND=gpu WITH_TESTS=1    # build the GPU backend + tests
mpirun -np 4 ./tests/gpu_equivalence_test   # GPU == CPU bit-exact check
```

---

## The CLI

Everything is a command-line option (the original Fortran `PARAMETER`/set-up
arrays are a plain struct filled from `argv`), so a run is described entirely
by the invocation:

```
mpirun -np <NPE> bin/3dmhd [options]
```

At least 2 MPI ranks are required (NPEZ >= 2). `--help` prints this summary.

### Geometry & process topology

| Option | Default | Description |
|---|---|---|
| `--npx N` `--npy N` `--npz N` | `504 504 2048` | Global grid points in x / y / z |
| `--npey N` | `8` | Processors along y; `npez = NPE / npey` is derived and must be >= 2 |
| `--ngrid {0,1,2}` | `1` | Grid-stretch function selector |
| `--ix 2` `--iy 2` `--ilap 2` | `2 2 2` | Horizontal/vertical stencil (ghost) half-widths |
| `--ixcon 0` `--iycon 0` `--izcon 0` `--itcon 1` `--ibcon 0` | | Boundary conditions (see constraints below) |

**Topology constraints** (validated at startup, otherwise the run aborts):

* `npey` divides both the MPI size `NPE` and `--npy`
* `npez = NPE / npey` divides `--npz` and is >= 2

So, for the default `504 x 504 x 2048` grid: 128 ranks -> `--npey 8` (npez=16),
8 ranks -> `--npey 4` (npez=2). Local arrays are `(504+ix) x (504/npey+iy) x
(2048/npez+ilap)`.

### Physics

| Option | Default | Description |
|---|---|---|
| `--re R` | `2` | Sound-speed Reynolds number |
| `--pr R` | `0.05` | Pseudo Prandtl number |
| `--theta R` | `0.25` | Temperature gradient |
| `--grav R` | `0.625` | Non-dimensional gravitational constant |
| `--rm R` | `0` | Magnetic Reynolds number |
| `--beta R` | `0` | Plasma beta |
| `--ampb R` `--bfh R` `--bzp R` | `0 0 0` | Magnetic layer amplitude / FWHM / depth |
| `--ampt R` | `0.005` | Amplitude of initial random temperature perturbations |
| `--gamma R` | `1.6666667` | Cp/Cv (single-precision 5/3, widened) |
| `--xmax R` `--ymax R` `--zmax R` | `20 20 40` | Domain extents |
| `--sff R` | `0.315` | Time-step safety factor (single-precision 0.315, widened) |
| `--pzp R` `--sigma R` | `0 0` | Stable-background transition depth / width |
| `--polys R` | `0` | Polytropic index of the lower stable layer |
| `--tp R` `--tc R` | `2 0` | Plume temperature / flux and its onset time scale |
| `--xp R` `--yp R` `--zp R` | `10 10 0` | Plume source position |
| `--hh R` | `1` | Tube width / plume spatial width |
| `--lrot 0\|1` | `0` | Rotation (`--ry`, `--ang`) |
| `--lmag 0\|1` | `0` | Magnetic fields |
| `--ntube N` | `0` | Tube model (0..5) initial condition |
| `--ncol N` `--nrow N` | `0 0` | Tube-array geometry |

### Grid-stretch coefficients

```
--xx1 R --xx2 R --yy1 R --yy2 R --zz1 R --zz2 R   (defaults -7 7  -7 7  -2 0.01)
--xa --xb --xc --xd      (and --ya/--yb/--yc/--yd, --za/--zb/--zc/--zd)
--xcent R --zcent R --rmax R --cmt R --a R --lambda R --vz0 R   (tube geometry)
```

### Execution

| Option | Default | Description |
|---|---|---|
| `--backend cpu\|gpu` | `cpu` | Pick the CPU or GPU time loop. A CPU-only binary aborts loudly if given `--backend gpu` |
| `--gpu ID` | `0` | CUDA device per rank (honours `CUDA_VISIBLE_DEVICES`) |
| `--ntotal N` | `1000` | Number of RK steps |
| `--nstep0 N` | `100` | Log / dump cadence (steps per checkpoint) |
| `--nstart N` | `0` | `0` = fresh STATIC start; restart is not yet ported |
| `--ncase N` `--ncasep N` | `1 1` | Case selectors |
| `--help` | | Print usage |

---

## Running

### CPU

```sh
mpirun -np 128 bin/3dmhd --npey 8 --ntotal 1000 --nstep0 100
```

### GPU (one rank per GPU)

```sh
# rank i gets GPU i (works for a single shared node; OMPI_COMM_WORLD_LOCAL_RANK)
mpirun -np 8 bash -c \
  'export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK; exec "$@"' _ \
  bin/3dmhd --backend gpu --npey 4 --ntotal 1000 --nstep0 100
```

### Slurm (ready-made scripts in `scripts/`)

```sh
make                  # CPU backend; or: make BACKEND=gpu
sbatch --ntasks=128   scripts/run_cpu.sbatch
sbatch --gres=gpu:8   scripts/run_gpu.sbatch
```

The scripts validate the process-grid/geometry decomposition, set up the Kez
OpenMPI environment, filter benign OpenMPI/UCX startup noise, pin one GPU per
rank (`CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK`), and report overall
wall time and step rate.  `run_cpu.sbatch` pins one core per rank (`1`
OpenMP thread each); for a hybrid run give each rank more cores
(`--cpus-per-task=8`) and `OMP_NUM_THREADS` scales with it.

---

## Output

- **Progress log** (stderr, merged into the slurm `.out`): one line per
  checkpoint with simulation time and wall-clock timing:
  ```
  NIT 100  DT 1.5e-06  TIMC 0.000155  UMACH 1.2e-05  WALL 320.1s (3.20s/step)
  ```
- **State dumps** (`gstate.<tag>.<rank>.<iter>`, self-describing binary, the
  golden format; see `golden/read_gstate.py`):
  - `gstate.start.*` — the initial condition (unless `DUMPMODE=none`)
  - `gstate.step.*.<NIT>` — checkpoints, controlled by the compile-time
    `DUMPMODE` (`make DUMPMODE=periodic|final|none`):
    - `periodic` — every `--nstep0` steps plus the final step (default)
    - `final` — only the last step (plus `gstate.start`)
    - `none` — no dump files at all; checkpoint lines are still logged

  Each dump is ~`NX*NY*NZ*27*8` bytes per rank (≈118 GB per checkpoint at the
  default 504x504x2048 grid) — use `DUMPMODE=final|none` for long runs.

---

## Reproducibility

The port is validated **bit-for-bit** against the original Fortran (an
instrumented golden build writes `gstate.*` snapshots that the gtest suite
compares with `EXPECT_DOUBLE_EQ`). Three things keep the numbers identical:

1. Single-precision `E00`/`FLOAT()` constants are reproduced with `f4()`
   (`numcompat.hpp`) before being widened to `double`.
2. Expression association is preserved exactly (left-to-right, the same
   groupings as the Fortran), including the `3dmhd` quirks (`A/(X*X)` vs
   `(A/X)/X`, etc.).
3. FMA contraction is disabled on both backends: `-ffp-contract=off` in the
   default `CXXFLAGS` and `-fmad=false` in the default `NVCCFLAGS`. Dropping
   either gives faster code at the cost of reproducible rounding; the golden
   match is only guaranteed with contraction disabled.

The gtest suite (`make WITH_TESTS=1 test`, `test_multi`, and the 4-rank
`tests/gpu_equivalence_test`) must pass bit-exact.

### Not yet ported

These options are recognized but abort with a clear message (the original
features are still being ported): `--lshr`, `--lrem`, `--rlax`, `--id`,
`--itcon 2|3`, non-periodic `--ixcon/--iycon`, `--lmag` (magnetic branch;
`--ampb` without `--lmag` also errors), and `--nstart` (restart).

---

## Packaging

A Kez recipe (`database/rast-3dmhd/latest.yaml`) wraps the make build, exposes
`BACKEND`, `CXX`, `NVCC`, `NVCC_ARCH`, `NVCCFLAGS`, `CXXFLAGS`, `DUMPMODE`,
`WITH_TESTS`/`GTEST_PREFIX` as user-configurable options, and installs via
`make install PREFIX=...`.
