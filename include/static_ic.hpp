// static_ic.hpp - initial conditions (port of 3dmhdsub.f STATIC + TUBE).
//
// STATIC builds the vertical stratification (T/R profile via the
// Bulirsch-Stoer ODE), distributes it over the z subdomains, optionally
// adds a magnetic layer or tube (LMAG), and seeds temperature perturbations
// with the RAN2 stream generated on rank 0 and fanned out over MPI exactely
// as the original does.
#pragma once

#include <mpi.h>

#include "fields.hpp"
#include "params.hpp"
#include "topology.hpp"

namespace r3d {

// Build the initial state (overwrites RU,RV,RW,TT,RO and B fields if LMAG).
// Assumes the grid metrics and RKAPA/DKAPA are already filled.
void static_ic(MPI_Comm comm, const Params& p, const Derived& d,
               const Topology& t, SimState& s);

}  // namespace r3d
