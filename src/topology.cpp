// topology.cpp - process-grid resolution and validation.
#include "topology.hpp"

#include <cstdio>
#include <cstdlib>

namespace r3d {

RankXY rank_to_xy(int rank, int npey) {
  RankXY r;
  r.mypey = rank % npey;
  r.mypez = rank / npey;
  return r;
}

namespace {
[[noreturn]] void die(const char* msg) {
  fprintf(stderr, "rast-3dmhd: %s\n", msg);
  exit(EXIT_FAILURE);
}
}  // namespace

Topology resolve_topology(int npe, int mype, const Params& p) {
  Topology t;
  t.npe = npe;
  t.mype = mype;

  if (p.npey <= 0) {
    // Auto-derive a square-ish grid with MPI_Dims_create semantics.
    int dims[2] = {0, 0};
    // Simple local equivalent of MPI_Dims_create(npe, 2, dims): find the
    // smallest factor >= sqrt(npe).
    for (int f = 1; f * f <= npe; ++f) {
      if (npe % f == 0) {
        dims[0] = f;  // y
        dims[1] = npe / f;  // z
      }
    }
    t.npey = dims[0];
    t.npez = dims[1];
  } else {
    t.npey = p.npey;
    if (npe % t.npey != 0) {
      fprintf(stderr,
              "rast-3dmhd: --npey %d does not divide MPI size %d\n",
              t.npey, npe);
      exit(EXIT_FAILURE);
    }
    t.npez = npe / t.npey;
  }

  if (t.npez < 2) {
    fprintf(stderr, "rast-3dmhd: the code requires NPEZ >= 2 (got %d with %d ranks and NPEY=%d)\n",
            t.npez, npe, t.npey);
    exit(EXIT_FAILURE);
  }
  if (p.npy % t.npey != 0) {
    fprintf(stderr, "rast-3dmhd: NPY=%d is not divisible by NPEY=%d\n", p.npy, t.npey);
    exit(EXIT_FAILURE);
  }
  if (p.npz % t.npez != 0) {
    fprintf(stderr, "rast-3dmhd: NPZ=%d is not divisible by NPEZ=%d\n", p.npz, t.npez);
    exit(EXIT_FAILURE);
  }

  RankXY r = rank_to_xy(mype, t.npey);
  t.mypey = r.mypey;
  t.mypez = r.mypez;
  return t;
}

}  // namespace r3d
