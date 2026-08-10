// communicate.hpp - MPI halo exchange (COMM_MPI / COMMUNICATE) and the
// horizontal mean used by optional diffusion/relaxation terms.
//
// The exchange mirrors 3dmhdsub.f COMM_MPI exactly:
//   vertical   : NX*NY*(ILAP/2)-thick slabs between neighbouring z ranks
//   horizontal : IY/2-thick y slabs between neighbouring y ranks plus the
//                periodic wrap, and the local x-periodic copy.
// Our arrays are x-major (x fastest), so vertical slabs are contiguous while
// y slabs are strided; both are handled through contiguous staging buffers.
#pragma once

#include <mpi.h>

#include "fields.hpp"
#include "params.hpp"
#include "topology.hpp"

namespace r3d {

    // Exchange halos of the 5 primary fields (and the B fields when lmag).
    void communicate(MPI_Comm comm, const Params& p, const Topology& t, SimState& s);

    // Exchange halos of one field (COMM_MPI).
    void comm_mpi(MPI_Comm comm, const Params& p, const Topology& t, Field& f);

    // Horizontal mean of a 3D field (HORIZONTAL_MEAN): varm(k) = average over
    // the interior x/y at each z, then reduced across y ranks.  For NGRID=0 the
    // sum is over the interior cells; the spline path for NGRID>0 is currently
    // disabled in the reference (it would terminate).
    void horizontal_mean(MPI_Comm comm, const Params& p, const Topology& t, const SimState& s,
                         const Field& f, double* varm);

}  // namespace r3d
