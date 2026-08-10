// fluxes_test - the FLUXES kernel validated against the golden Fortran
// reference.  Loads the golden pre-loop state and compares the kernel
// output against the golden snapshot taken right after the first CALL FLUXES
// inside STEP.
#include "fluxes.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "golden_util.hpp"
#include "gstate_reader.hpp"

using namespace r3d;
using namespace r3dtest;

namespace {
    void expect_interior_match(const Field& f, const std::vector<double>& g, const Params& p,
                               const char* name) {
        double err = max_err_interior(f, g, p.ix, p.iy, p.ilap);
        EXPECT_NEAR(err, 0.0, 0.0) << name << " interior error";
    }
}  // namespace

TEST(Fluxes, HydroMatchesGoldenExactly) {
    Params p       = golden_params();  // default hydro: no mag / rotation etc.
    const int mype = 0;                // golden rank 0 (mypey=0, mypez=0)
    Topology t     = golden_topo(mype);
    Derived d      = derive(p, t.npe, t.npey, t.npez);
    SimState s(d);

    Gstate pre = read_gstate(golden_dir() + "/gstate.preloop.000.0000");
    load_state(s, pre);

    Gstate g = read_gstate(golden_dir() + "/gstate.flux1.000.0000");

    fluxes(p, d, t, s.rkapa.data(), s.dkapa.data(), s);

    // The output fluxes are written over the interior cells.
    expect_interior_match(s.fu, g.f3d[Gstate::idx3d("fu")], p, "fu");
    expect_interior_match(s.fv, g.f3d[Gstate::idx3d("fv")], p, "fv");
    expect_interior_match(s.fw, g.f3d[Gstate::idx3d("fw")], p, "fw");
    expect_interior_match(s.fr, g.f3d[Gstate::idx3d("fr")], p, "fr");
    expect_interior_match(s.ft, g.f3d[Gstate::idx3d("ft")], p, "ft");
    // RO = 1/RHO and the velocities.
    expect_interior_match(s.ro, g.f3d[Gstate::idx3d("ro")], p, "ro");
    expect_interior_match(s.uu, g.f3d[Gstate::idx3d("uu")], p, "uu");
    expect_interior_match(s.vv, g.f3d[Gstate::idx3d("vv")], p, "vv");
    expect_interior_match(s.ww, g.f3d[Gstate::idx3d("ww")], p, "ww");
    // WW1..WW3 are scratch; compare interiors for completeness.
    expect_interior_match(s.ww1, g.f3d[Gstate::idx3d("ww1")], p, "ww1");
    expect_interior_match(s.ww2, g.f3d[Gstate::idx3d("ww2")], p, "ww2");
    expect_interior_match(s.ww3, g.f3d[Gstate::idx3d("ww3")], p, "ww3");
}

TEST(Fluxes, UniformStateYieldsZeroFluxes) {
    // With a uniformly constant state (no gradients in any direction) and no
    // rotation/diffusion, the horizontal/advective/diffusive fluxes vanish;
    // only the vertical force FW=GRAV*RHO (buoyancy) remains.
    Params p   = golden_params();
    Topology t = golden_topo(0);
    Derived d  = derive(p, t.npe, t.npey, t.npez);
    SimState s(d);
    s.ru.fill(0.0);
    s.rv.fill(0.0);
    s.rw.fill(0.0);
    s.ro.fill(1.0);
    s.tt.fill(1.0);

    fluxes(p, d, t, s.rkapa.data(), s.dkapa.data(), s);

    std::vector<double> zero(s.d.nx * s.d.ny * s.d.nz, 0.0);
    expect_interior_match(s.fu, zero, p, "fu");
    expect_interior_match(s.fv, zero, p, "fv");
    expect_interior_match(s.fr, zero, p, "fr");
    expect_interior_match(s.ft, zero, p, "ft");
    // FW should equal GRAV*RHO everywhere in the interior.
    std::vector<double> fw_grav(s.d.nx * s.d.ny * s.d.nz, p.grav);
    expect_interior_match(s.fw, fw_grav, p, "fw");
}
