// communicate_test - halo exchange + horizontal mean under a real MPI run.
// Run with: mpirun -np 4 ./tests/communicate_test  (golden geometry below).
//
// The invariant tested: if every cell holds a *globally periodic* function
// evaluated at that cell's global (x,y,z), then after COMMUNICATE every cell
// (including ghost/border layers) must still equal that function.  Any index
// error in the exchange breaks the equality.
#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "communicate.hpp"
#include "fields.hpp"
#include "params.hpp"
#include "topology.hpp"

using namespace r3d;

namespace {

// Golden geometry: NPEY=2, NPEZ=2, NPX=16, NPY=16, NPZ=32.
Params golden_params() {
  Params p;
  p.npx = 16; p.npy = 16; p.npz = 32;
  p.npey = 2;
  return p;
}

// Map a local cell to its global periodic index.
int gx_of(const Params& p, int i) {
  int period = p.npx;
  return ((i + period - 1) % period + period) % period;  // ix/2 = 1
}
int gy_of(const Params& p, const Topology& t, int j) {
  int period = p.npy;
  int g = j + period - 1 - t.mypey * (p.npy / t.npey);
  return ((g % period) + period) % period;
}
int gz_of(const Params& p, const Topology& t, int k) {
  int period = p.npz;
  int g = k + period - 1 - t.mypez * (p.npz / t.npez);
  return ((g % period) + period) % period;
}

double truth_fn(double gx, double gy, double gz) {
  return std::sin(0.1 * gx + 0.2 * gy + 0.3 * gz);
}

}  // namespace

TEST(Communicate, PeriodicFieldSurvivesExchange) {
  int npes, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  Params p = golden_params();
  Topology t = resolve_topology(npes, rank, p);
  Derived d = derive(p, npes, t.npey, t.npez);
  SimState s(d);

  // Periodic function of global coordinates, evaluated on the local grid.
  for (int k = 0; k < d.nz; ++k)
    for (int j = 0; j < d.ny; ++j)
      for (int i = 0; i < d.nx; ++i)
        s.ru.at(i, j, k) =
            truth_fn(gx_of(p, i), gy_of(p, t, j), gz_of(p, t, k));

  comm_mpi(MPI_COMM_WORLD, p, t, s.ru);

  double maxerr = 0.0;
  for (int k = 0; k < d.nz; ++k)
    for (int j = 0; j < d.ny; ++j)
      for (int i = 0; i < d.nx; ++i) {
        double want =
            truth_fn(gx_of(p, i), gy_of(p, t, j), gz_of(p, t, k));
        double err = std::fabs(s.ru.at(i, j, k) - want);
        if (err > maxerr) maxerr = err;
      }
  EXPECT_LT(maxerr, 1e-12) << "rank " << rank;
}

TEST(Communicate, AllPrimaryFieldsCommunicate) {
  int npes, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  Params p = golden_params();
  Topology t = resolve_topology(npes, rank, p);
  Derived d = derive(p, npes, t.npey, t.npez);
  SimState s(d);

  auto periodic_fill = [&](Field& f, double mult) {
    for (int k = 0; k < d.nz; ++k)
      for (int j = 0; j < d.ny; ++j)
        for (int i = 0; i < d.nx; ++i)
          f.at(i, j, k) = mult *
                        truth_fn(gx_of(p, i), gy_of(p, t, j), gz_of(p, t, k));
  };
  periodic_fill(s.ru, 1.0);
  periodic_fill(s.rv, 2.0);
  periodic_fill(s.rw, 3.0);
  periodic_fill(s.ro, 4.0);
  periodic_fill(s.tt, 5.0);
  if (p.lmag) {
    periodic_fill(s.bx, 6.0);
    periodic_fill(s.by, 7.0);
    periodic_fill(s.bz, 8.0);
  }

  communicate(MPI_COMM_WORLD, p, t, s);

  auto check = [&](const Field& f, double mult, const char* name) {
    double maxerr = 0.0;
    for (int k = 0; k < d.nz; ++k)
      for (int j = 0; j < d.ny; ++j)
        for (int i = 0; i < d.nx; ++i) {
          double want = mult *
                        truth_fn(gx_of(p, i), gy_of(p, t, j), gz_of(p, t, k));
          maxerr = std::max(maxerr, std::fabs(f.at(i, j, k) - want));
        }
    EXPECT_LT(maxerr, 1e-12) << name << " rank " << rank;
  };
  check(s.ru, 1.0, "ru");
  check(s.rv, 2.0, "rv");
  check(s.rw, 3.0, "rw");
  check(s.ro, 4.0, "ro");
  check(s.tt, 5.0, "tt");
}

TEST(Communicate, HorizontalMeanOfSinusoidalVerticalProfile) {
  int npes, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  Params p = golden_params();
  p.ngrid = 0;  // HORIZONTAL_MEAN only supports the uniform grid
  Topology t = resolve_topology(npes, rank, p);
  Derived d = derive(p, npes, t.npey, t.npez);
  SimState s(d);

  // Field constant in x,y but varying vertically: the horizontal mean at
  // each k must equal that value everywhere.
  for (int k = 0; k < d.nz; ++k)
    for (int j = 0; j < d.ny; ++j)
      for (int i = 0; i < d.nx; ++i)
        s.uu.at(i, j, k) = 1.0 + 0.5 * std::cos(0.1 * k);

  std::vector<double> varm(d.nz);
  horizontal_mean(MPI_COMM_WORLD, p, t, s, s.uu, varm.data());
  for (int k = 0; k < d.nz; ++k) {
    double want = 1.0 + 0.5 * std::cos(0.1 * k);
    EXPECT_NEAR(varm[k], want, 1e-12) << "rank " << rank << " k " << k;
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  MPI_Finalize();
  return rc;
}
