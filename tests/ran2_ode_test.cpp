// ran2_ode_test - RAN2 and the Bulirsch-Stoer static-profile integration,
// validated against golden values from golden/fortran/ran2_ode_ref.f.
#include <gtest/gtest.h>

#include <cmath>

#include "ode.hpp"
#include "params.hpp"
#include "ran2.hpp"

using namespace r3d;

#include "_ran2_ref.inc"

TEST(Ran2, GoldenSequence) {
    Ran2State st;
    st.idum    = -62659;  // seed (mirrors IDUM=-062659 in the original)
    size_t ref = 0;
    for (int i = 1; i <= 30000; ++i) {
        double last = ran2(st);
        if (i % 1000 == 0) {
            EXPECT_DOUBLE_EQ(last, kRan2Ref[ref].v) << "draw " << i;
            ++ref;
        }
    }
    EXPECT_EQ(ref, 30u);
}

TEST(Ran2, SequenceIsDeterministic) {
    Ran2State a, b;
    a.idum = -62659;
    b.idum = -62659;
    for (int i = 0; i < 2000; ++i) {
        EXPECT_DOUBLE_EQ(ran2(a), ran2(b));
    }
}

namespace {
    Params golden_params() {
        Params p;
        p.npx  = 16;
        p.npy  = 16;
        p.npz  = 32;
        p.npey = 2;
        return p;
    }
}  // namespace

TEST(Ode, GoldenStaticProfile) {
    Params p      = golden_params();
    const int npz = 32;
    std::vector<double> T(npz), R(npz);
    static_profile(p, npz, T.data(), R.data());

    // Reference values from ran2_ode_ref.f (with the golden defaults).
    EXPECT_DOUBLE_EQ(T[0], 1.0);
    EXPECT_DOUBLE_EQ(T[3], 3.21832888589760424);   // T(4)
    EXPECT_DOUBLE_EQ(T[9], 6.00075810678302801);   // T(10)
    EXPECT_DOUBLE_EQ(T[19], 8.71388943224966717);  // T(20)
    EXPECT_DOUBLE_EQ(T[31], 10.9999998343435870);  // T(32)
    EXPECT_DOUBLE_EQ(R[31], 36.4828718697596557);  // R(32)
}

TEST(Ode, ProfileMonotonic) {
    Params p = golden_params();
    std::vector<double> T(32), R(32);
    static_profile(p, 32, T.data(), R.data());
    for (int k = 1; k < 32; ++k) {
        EXPECT_GT(T[k], T[k - 1]);
        EXPECT_GT(R[k], R[k - 1]);
    }
}
