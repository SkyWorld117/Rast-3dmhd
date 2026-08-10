// fluxes.hpp - the FLUXES kernel: time derivatives of the 8 primary fields.
//
// Port of 3dmhdsub.f FLUXES.  Every loop sweep is preserved 1:1 (with
// Fortran 1-based -> 0-based index mapping), so the result matches the
// golden Fortran reference.  A few Fortran loops range over the *full*
// local array (e.g. WW1=RO*TT, RO=1/RO, UU=RU*RO) including ghost cells;
// those are reproduced as full-array sweeps because later stencils read the
// ghost values.
#pragma once

#include "fields.hpp"
#include "params.hpp"
#include "topology.hpp"

namespace r3d {

    // Compute FU,FV,FW,FR,FT constraints (and WW1..WW3 as B fluxes when LMAG)
    // from the current physical state RO,RV,RW,TT,RO(,B fields).  Overwrites
    // RO with 1/RO and UU,VV,WW with velocities (like the Fortran), and uses
    // WW1..WW3 as scratch.
    void fluxes(const Params& p, const Derived& d, const Topology& t, const double rkapa[] /*nz*/,
                const double dkapa[] /*nz*/, SimState& s);

}  // namespace r3d
