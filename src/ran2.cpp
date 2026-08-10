// ran2.cpp - RAN2 (Numerical Recipes).
#include "ran2.hpp"

#include "numcompat.hpp"

namespace r3d {

    namespace {
        constexpr int kM  = 714025;
        constexpr int kIA = 1366;
        constexpr int kIC = 150889;
        // RM = 1.0E00/714025.0E00 : single-precision division in the original
        // (both literals are E00), then widened into REAL*8.
        const double kRM = f4(1.0f / 714025.0f);
    }  // namespace

    double ran2(Ran2State& s) {
        if (s.idum < 0) {
            s.idum = (kIC - s.idum) % kM;
            for (int j = 0; j < 97; ++j) {
                s.idum   = (kIA * s.idum + kIC) % kM;
                s.iir[j] = s.idum;
            }
            s.idum = (kIA * s.idum + kIC) % kM;
            s.iiy  = s.idum;
        }
        int j = 1 + (97 * s.iiy) / kM;  // 1-based -> subtract 1 for 0-based array
        if (j > 97 || j < 1) {
            // RAN2: Error -> terminates hard (mirrors MPI_FINALIZE + STOP).
            __builtin_trap();
        }
        s.iiy         = s.iir[j - 1];
        double result = (double) s.iiy * kRM;
        s.idum        = (kIA * s.idum + kIC) % kM;
        s.iir[j - 1]  = s.idum;
        return result;
    }

}  // namespace r3d
