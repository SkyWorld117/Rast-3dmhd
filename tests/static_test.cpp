// static_test - STATIC initial conditions validated against the golden
// Fortran.  Run with: mpirun -np 4 ./tests/static_test
//
// Compares the stratification (TT/RO rows), the zeroed velocities, and the
// RAN2 temperature perturbation pattern (WW1) against gstate.static.00X.
// Only interior x/y cells are compared at the pre-COMMUNICATE snapshot
// (the original leaves the horizontal ghosts uninitialised there).
#include <gtest/gtest.h>
#include <mpi.h>

#include <vector>

#include "golden_util.hpp"
#include "gstate_reader.hpp"
#include "grid.hpp"
#include "ode.hpp"
#include "params.hpp"
#include "static_ic.hpp"

using namespace r3d;
using namespace r3dtest;

namespace {
// Compare a field over interior x/y and ALL k.
double max_err_cols(const Field& f, const std::vector<double>& g,
                    const Params& p, int ilap) {
  const int nx = f.nx(), ny = f.ny(), nz = f.nz();
  double m = 0.0;
  for (int k = 0; k < nz; ++k)
    for (int j = 1; j <= ny - p.iy; ++j)
      for (int i = 1; i <= nx - p.ix; ++i)
        m = std::max(m, std::fabs(f.at(i, j, k) - g[(long)k * nx * ny + (long)j * nx + i]));
  return m;
}
}  // namespace

TEST(Static, MatchesGolden) {
  int npes, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (npes != 4) GTEST_SKIP() << "requires 4 ranks";

  Params p = golden_params();
  Topology t = resolve_topology(npes, rank, p);
  Derived d = derive(p, npes, t.npey, t.npez);
  SimState s(d);
  build_grid_metrics(p, t, s);
  // RKAPA=1, DKAPA=0 for the default PZP == 0.
  for (auto& v : s.rkapa) v = 1.0;
  for (auto& v : s.dkapa) v = 0.0;

  static_ic(MPI_COMM_WORLD, p, d, t, s);

  std::string tag = "00" + std::to_string(rank);
  Gstate g = read_gstate(golden_dir() + "/gstate.static." + tag + ".0000");
  ASSERT_EQ(g.mype, rank);

  EXPECT_NEAR(max_err_cols(s.ru, g.f3d[Gstate::idx3d("ru")], p, p.ilap), 0.0, 1e-15)
      << "ru rank " << rank;
  EXPECT_NEAR(max_err_cols(s.rv, g.f3d[Gstate::idx3d("rv")], p, p.ilap), 0.0, 1e-15)
      << "rv rank " << rank;
  EXPECT_NEAR(max_err_cols(s.rw, g.f3d[Gstate::idx3d("rw")], p, p.ilap), 0.0, 1e-15)
      << "rw rank " << rank;
  EXPECT_NEAR(max_err_cols(s.tt, g.f3d[Gstate::idx3d("tt")], p, p.ilap), 0.0, 1e-15)
      << "tt rank " << rank;
  EXPECT_NEAR(max_err_cols(s.ro, g.f3d[Gstate::idx3d("ro")], p, p.ilap), 0.0, 1e-15)
      << "ro rank " << rank;

  // The perturbation pattern (WW1) over the covered cells.
  double ww1err = 0.0;
  const int nx = d.nx, ny = d.ny, nz = d.nz;
  for (int k = p.ilap / 2; k <= nz - 1 - p.ilap / 2; ++k)
    for (int j = p.iy / 2; j <= ny - p.iy / 2 - 1; ++j)
      for (int i = p.ix / 2; i <= nx - p.ix / 2 - 1; ++i)
        ww1err = std::max(
            ww1err, std::fabs(s.ww1.at(i, j, k) -
                              g.f3d[Gstate::idx3d("ww1")][(long)k * nx * ny +
                                                          (long)j * nx + i]));
  EXPECT_NEAR(ww1err, 0.0, 1e-15) << "ww1 perturbation rank " << rank;
  std::vector<double> T(p.npz), R(p.npz);
  static_profile(p, p.npz, T.data(), R.data());
  EXPECT_DOUBLE_EQ(s.tb, T[p.npz - 1]) << "tb rank " << rank;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  MPI_Finalize();
  return rc;
}
