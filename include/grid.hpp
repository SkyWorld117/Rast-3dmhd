// grid.hpp - grid generation: MKGRID + X/Y/ZJACOBI.
//
// These are direct ports of the Fortran routines in 3dmhdsub.f.  MKGRID is
// a pure function of (params, scode, icoord); the Jacobi routines fill a
// rank-local domain and therefore also take the processor coordinates.
#pragma once

#include "params.hpp"
#include "topology.hpp"

namespace r3d {

    // Map a computational coordinate scode in [0,1] to the physical coordinate
    // and its first two derivatives, for coordinate direction icoord (1=x,2=y,3=z).
    // Selects on params.ngrid: 0 uniform, 1 arctan, 2 two-arctan.
    void mkgrid(const Params& p, double scode, int icoord, double& sphys, double& dsds,
                double& d2sds2);

    // Fill the rank-local x/y/z Jacobian metrics into the given buffers.
    // Buffers are 1D of lengths nx / ny / nz (the full local extents); the
    // interior (including border cells written here) is [1, n-dim) plus ghosts.
    void xjacobi(const Params& p, int nx, double* exx, double* dxxdx, double* d2xxdx2, double* ddx);
    void yjacobi(const Params& p, int mypey, int npey, int ny, int npy, int nry, double* wyy,
                 double* dyydy, double* d2yydy2, double* ddy);
    void zjacobi(const Params& p, int mypez, int npez, int nz, int npz, int nrz, double* zee,
                 double* dzzdz, double* d2zzdz2, double* ddz);

    // Convenience: compute all three into a SimState's metric arrays.
    struct SimState;
    void build_grid_metrics(const Params& p, const Topology& t, SimState& s);

}  // namespace r3d
