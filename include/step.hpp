// step.hpp - one RK3 time step (port of 3dmhdsub.f STEP).
//
// STEP computes the pointwise-minimum explicit timestep, the maximum Mach
// number, and advances the state through three low-storage Wray RK3
// substeps using FLUXES, applying BCON and the halo exchange between
// stages.  Mirrors the Fortran exactly (ISW=1 pointwise timestep branch).
#pragma once

#include <mpi.h>

#include "fields.hpp"
#include "params.hpp"
#include "topology.hpp"

namespace r3d {

// Advance the state by one step; updates s.dt, s.timc, s.timt, s.nit.
void step(MPI_Comm comm, const Params& p, const Derived& d, const Topology& t,
          SimState& s);

// Advance by exactly nstages Runge-Kutta substages (1..3); used by tests to
// compare intermediate RK states against the golden reference.
void step(int nstages, MPI_Comm comm, const Params& p, const Derived& d,
          const Topology& t, SimState& s);

}  // namespace r3d
