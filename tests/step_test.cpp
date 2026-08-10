// step_test - one full RK3 STEP validated against the golden Fortran.
// Run with: mpirun -np 4 ./tests/step_test
//
// Each rank loads its golden pre-loop state, advances by one step, and
// compares the resulting primary fields + dt against the golden step-1
// dump (gstate.step.00X.0001).
#include <gtest/gtest.h>
#include <mpi.h>

#include <string>
#include <vector>

#include "golden_util.hpp"
#include "gstate_reader.hpp"
#include "ode.hpp"
#include "params.hpp"
#include "fluxes.hpp"
#include "step.hpp"

using namespace r3d;
using namespace r3dtest;

TEST(Step, OneRankMatchesGolden) {
  int npes, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (npes != 4) {
    GTEST_SKIP() << "step_test requires exactly 4 ranks (golden geometry)";
  }

  Params p = golden_params();
  Topology t = resolve_topology(npes, rank, p);
  Derived d = derive(p, npes, t.npey, t.npez);
  SimState s(d);

  std::string tag = "00" + std::to_string(rank);
  // Lower-boundary temperature from the static profile (set by STATIC in a
  // full run; reconstructed here since we load the golden pre-loop state).
  std::vector<double> T(p.npz), R(p.npz);
  static_profile(p, p.npz, T.data(), R.data());
  s.tb = T[p.npz - 1];

  Gstate pre = read_gstate(golden_dir() + "/gstate.preloop." + tag + ".0000");
  ASSERT_EQ(pre.mype, rank);
  load_state(s, pre);

  step(MPI_COMM_WORLD, p, d, t, s);

  Gstate g = read_gstate(golden_dir() + "/gstate.step." + tag + ".0001");
  EXPECT_EQ(g.nit, 1);
  (void)0;

  // The global dt must match the golden scalar.
  EXPECT_DOUBLE_EQ(s.dt, g.scalars[0]) << "rank " << rank;
  EXPECT_DOUBLE_EQ(s.timc, g.scalars[1]) << "rank " << rank;

  // Primary fields (interiors) must match exactly.
  auto expect_match = [&](const Field& fld, const char* name) {
    double err = max_err_interior(fld, g.f3d[Gstate::idx3d(name)], p.ix, p.iy,
                                  p.ilap);
    // Locate the first differing interior cell for diagnostics.
    if (err != 0.0) {
      const int nx = d.nx, ny = d.ny, nz = d.nz;
      bool shown = false;
      for (int k = p.ilap / 2; k <= nz - 1 - p.ilap / 2 && !shown; ++k)
        for (int j = 1; j <= ny - p.iy && !shown; ++j)
          for (int i = 1; i <= nx - p.ix; ++i) {
            const long idx = (long)k * nx * ny + (long)j * nx + i;
            const double e = fld.at(i, j, k) - g.f3d[Gstate::idx3d(name)][idx];
            if (e != 0.0) {
              fprintf(stderr,
                      "STEP_LOCATE %s rank %d (i=%d,j=%d,k=%d) diff=%.3e "
                      "mine=%.17g golden=%.17g\n",
                      name, rank, i, j, k, e, fld.at(i, j, k),
                      g.f3d[Gstate::idx3d(name)][idx]);
              shown = true;
            }
          }
    }
    EXPECT_NEAR(err, 0.0, 0.0) << name << " rank " << rank;
  };
  expect_match(s.ru, "ru");
  expect_match(s.rv, "rv");
  expect_match(s.rw, "rw");
  expect_match(s.ro, "ro");
  expect_match(s.tt, "tt");
}





// Pre-viscous FLUXES output vs golden ft2.

// Full stage-2 FLUXES vs golden flux2.
TEST(Step, FullFlux2MatchesGolden) {
  int npes, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (npes != 4) GTEST_SKIP() << "requires 4 ranks";
  Params p = golden_params();
  Topology t = resolve_topology(npes, rank, p);
  Derived d = derive(p, npes, t.npey, t.npez);
  SimState s(d);
  std::vector<double> T(p.npz), R(p.npz);
  static_profile(p, p.npz, T.data(), R.data());
  s.tb = T[p.npz - 1];
  std::string tag = "00" + std::to_string(rank);
  Gstate pre = read_gstate(golden_dir() + "/gstate.preloop." + tag + ".0000");
  load_state(s, pre);
  step(1, MPI_COMM_WORLD, p, d, t, s);
  fluxes(p, d, t, s.rkapa.data(), s.dkapa.data(), s);
  Gstate g = read_gstate(golden_dir() + "/gstate.flux2." + tag + ".0000");
  auto check = [&](const Field& fld, const char* name) {
    double err = max_err_interior(fld, g.f3d[Gstate::idx3d(name)], p.ix, p.iy, p.ilap);
    if (err != 0.0) {
      double best=0; int bi=-1,bj=-1,bk=-1;
      for (int kk=p.ilap/2; kk<=d.nz-1-p.ilap/2; ++kk)
        for (int jj=1; jj<=d.ny-p.iy; ++jj)
          for (int ii=1; ii<=d.nx-p.ix; ++ii) {
            long idx=(long)kk*d.nx*d.ny+(long)jj*d.nx+ii;
            double e=std::fabs(fld.at(ii,jj,kk)-g.f3d[Gstate::idx3d(name)][idx]);
            if(e>best){best=e;bi=ii;bj=jj;bk=kk;}
          }
      long id=(long)bk*d.nx*d.ny+(long)bj*d.nx+bi;
      fprintf(stderr, "FX2 %s rank %d (i=%d j=%d k=%d) e=%.3e mine=%.17g golden=%.17g\n",
              name, rank, bi,bj,bk,best, fld.at(bi,bj,bk), g.f3d[Gstate::idx3d(name)][id]);
    }
    EXPECT_NEAR(err, 0.0, 0.0) << "fullflux2 " << name << " rank " << rank;
  };
  check(s.fu, "fu"); check(s.fv, "fv"); check(s.fw, "fw");
  check(s.fr, "fr"); check(s.ft, "ft");
}

// Post-momentum-viscous FLUXES vs golden fvm.

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  MPI_Finalize();
  return rc;
}
