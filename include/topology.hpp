// topology.hpp - 2D processor-grid resolution.
//
// The original Fortran hardcoded NPEY/NPEZ in 3dmhdparam.f and computed
// MYPEY=MOD(MYPE,NPEY), MYPEZ=MYPE/NPEY.  Here the grid is *parsed*:
// - the user supplies the y-processor count (--npey) on the command line,
// - the z-processor count is derived so that npey*npez == MPI comm size,
// - divisibility of NPY by NPEY and NPZ by NPEZ is validated at startup,
// - NPEY == 1 is allowed for single-row topologies (but the code still
//   requires NPEZ >= 2, mirroring the original constraint).
#pragma once

#include "params.hpp"

namespace r3d {

// Pure mapping: MPI rank -> (mypey, mypez) for the given grid.  Rank layout
// matches the original: mypey = rank % npey, mypez = rank / npey.
struct RankXY {
  int mypey;
  int mypez;
};
RankXY rank_to_xy(int rank, int npey);

// Resolve the process grid for a communicator of 'npe' ranks.
//   npey_in == 0  : auto-derive a square-ish grid with MPI_Dims_create
//   npey_in  > 0  : use it, npez = npe / npey must divide evenly
// Validates: npey divides npy, npez divides npz, npez >= 2.
// On error this terminates the program (ERROR macro semantics).
struct Topology {
  int npe;          // communicator size
  int npey, npez;   // resolved processor grid
  int mype;         // rank
  int mypey, mypez; // rank coordinates
};

Topology resolve_topology(int npe, int mype, const Params& p);

}  // namespace r3d
