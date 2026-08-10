// ran2.hpp - RAN2 uniform random generator (Numerical Recipes, p. 191).
//
// Port of the Fortran RAN2 from 3dmhdsub.f.  The Fortran 'IDUM' argument is
// passed by reference and modified by the generator, becoming positive after
// the first (seeding) call; we mirror that by keeping the seed in the state
// and seeding automatically on the first call.  The sequence depends on int32
// arithmetic and on RM = 1.0E00/714025.0E00 which the original computes in
// single precision (see numcompat.hpp).
#pragma once

namespace r3d {

    struct Ran2State {
        int idum    = 0;  // negative -> seed on next call, then updated in place
        int iiy     = 0;
        int iir[97] = {0};
    };

    // Draw the next deviate in (0,1).  On the first call (when idum < 0) the
    // generator is seeded from idum.  Returns a double in (0,1).
    double ran2(Ran2State& s);

}  // namespace r3d
