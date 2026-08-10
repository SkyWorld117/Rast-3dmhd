// bcon.hpp - boundary conditions (port of 3dmhdsub.f BCON).
//
// Applied after each Runge-Kutta substep update: sets stress-free,
// impenetrable velocity boundaries and the temperature / temperature-flux
// boundary conditions on the upper (k=ilap/2) and lower (k=nz-1-ilap/2)
// planes of the first/last z ranks.
#pragma once

#include "fields.hpp"
#include "params.hpp"
#include "topology.hpp"

namespace r3d {

    void bcon(const Params& p, const Derived& d, const Topology& t, SimState& s);

}  // namespace r3d
