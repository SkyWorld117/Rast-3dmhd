// gpu.hpp - device-resident simulation state and CUDA kernel entry points.
//
// The whole SimState (all 3D fields + 1D metrics) lives on the GPU; host
// copies are made only at upload time and at dump/reduction points.  Kernels
// in gpu_kernels.cu mirror the CPU reference sweep-for-sweep so the results
// are bit-identical (the CPU port is golden-validated).
#pragma once

#include <mpi.h>

#include "fields.hpp"
#include "params.hpp"
#include "topology.hpp"

namespace r3d {

// Device mirror image of SimState: raw device arrays (x-fastest layout).
struct GpuSim {
  int nx = 0, ny = 0, nz = 0;
  double *ru = nullptr, *rv, *rw, *ro, *tt;
  double *uu = nullptr, *vv, *ww;
  double *fu = nullptr, *fv, *fw, *fr, *ft;
  double *zru = nullptr, *zrv, *zrw, *zro, *ztt;
  double *ww1 = nullptr, *ww2, *ww3;
  double *bx = nullptr, *by, *bz, *zbx, *zby, *zbz;
  double *exx = nullptr, *dxxdx, *d2xxdx2, *ddx;
  double *wyy = nullptr, *dyydy, *d2yydy2, *ddy;
  double *zee = nullptr, *dzzdz, *d2zzdz2, *ddz;
  double *rkapa = nullptr, *dkapa;
};

// Allocate device buffers sized from d, upload all host state, register the
// object for use by step_gpu().
void gpu_alloc(GpuSim& g, const Derived& d);
void gpu_download(SimState& s, GpuSim& g);       // device -> host (all fields)
void gpu_upload(GpuSim& g, const SimState& s);   // host -> device (all fields)
void gpu_free(GpuSim& g);

// Advance one step entirely on the device (CUDA-aware MPI for the halo
// exchange and reductions).  Equivalent to step() up to the exact double
// arithmetic of the kernels (same expressions, so bit-identical).
void step_gpu(MPI_Comm comm, const Params& p, const Derived& d, const Topology& t,
              GpuSim& g, SimState& scalars);

}  // namespace r3d
