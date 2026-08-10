// gpu_kernels.hpp - device-side kernel declarations shared between the .cu
// implementations and the host driver (gpu.cpp, compiled by nvcc).
#pragma once

namespace r3d {

// Mirrors the Derived scalars used by the CPU sweep loops.
struct K {
  int nx, ny, nz;
  int i1, i2, j1, j2, k1, k2;   // interior (inclusive)
  double c13, c23, c43, ore, ocv, repr, grav;
  double hx, h2x, hy, h2y, hz, h2z;
  double omx, omz;
  double gamma, re, cv;
  double xp, yp, hh;
  int li, lj, lk;               // ilap/2, iy/2, ix/2
};

}  // namespace r3d
