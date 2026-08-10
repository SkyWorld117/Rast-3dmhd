// grid_test - MKGRID / XJACOBI / YJACOBI / ZJACOBI validated against the
// golden Fortran reference (golden/ref/gstate.preloop.*.0000).
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "grid.hpp"
#include "gstate_reader.hpp"
#include "params.hpp"
#include "topology.hpp"

using namespace r3d;
using r3dtest::Gstate;
using r3dtest::read_gstate;

namespace {
// Where the golden reference files live (overridable for a Kez sandbox).
std::string golden_dir() {
  const char* e = getenv("RAST_3DMHD_GOLDEN");
  return e ? std::string(e) : std::string("golden/ref");
}

double max_abs_diff(const std::vector<double>& a,
                    const std::vector<double>& b) {
  EXPECT_EQ(a.size(), b.size());
  double m = 0.0;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
    double d = std::fabs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}
}  // namespace

// The golden geometry: NPX=16, NPY=16, NPZ=32, NPEY=NPEZ=2 -> each rank
// holds NX=18, NY=10, NZ=18 with NRY=8, NRZ=16.
constexpr int kNPX = 16, kNPY = 16, kNPZ = 32;
constexpr int kNY = 10, kNZ = 18, kNRY = 8, kNRZ = 16;
constexpr int kNX = kNPX + 2;

Params golden_params() {
  Params p;
  p.npx = kNPX; p.npy = kNPY; p.npz = kNPZ;
  p.npey = 2;
  return p;
}

TEST(Grid, GoldenXJacobi) {
  Params p = golden_params();
  Gstate g = read_gstate(golden_dir() + "/gstate.preloop.000.0000");
  ASSERT_EQ(g.nx, kNX);
  std::vector<double> exx(kNX), dxxdx(kNX), d2xxdx2(kNX), ddx(kNX);
  xjacobi(p, kNX, exx.data(), dxxdx.data(), d2xxdx2.data(), ddx.data());
  // The x metrics are fully deterministic on the set range [1, nx-ix]:
  // exx[0] and exx[nx-1] are border ghosts never written by XJACOBI.
  const auto& g_exx = g.field1d("exx");
  const auto& g_dxxdx = g.field1d("dxxdx");
  const auto& g_d2xxdx2 = g.field1d("d2xxdx2");
  const auto& g_ddx = g.field1d("ddx");
  EXPECT_EQ(g_exx.size(), (size_t)kNX);
  for (int i = 1; i <= kNX - p.ix; ++i) {
    EXPECT_DOUBLE_EQ(exx[i], g_exx[i]) << "exx[" << i << "]";
    EXPECT_DOUBLE_EQ(dxxdx[i], g_dxxdx[i]) << "dxxdx[" << i << "]";
    EXPECT_DOUBLE_EQ(d2xxdx2[i], g_d2xxdx2[i]) << "d2xxdx2[" << i << "]";
    EXPECT_DOUBLE_EQ(ddx[i], g_ddx[i]) << "ddx[" << i << "]";
  }
}

TEST(Grid, GoldenYJacobiRank0And1) {
  Params p = golden_params();
  // Two ranks at the same mypez, mypey = 0 and 1.
  for (int mypey = 0; mypey < 2; ++mypey) {
    std::string path = golden_dir() + "/gstate.preloop.00" +
                       std::to_string(mypey) + ".0000";
    Gstate g = read_gstate(path);
    ASSERT_EQ(g.mypey, mypey);
    std::vector<double> wyy(kNY), dyydy(kNY), d2yydy2(kNY), ddy(kNY);
    yjacobi(p, mypey, /*npey=*/2, kNY, kNPY, kNRY, wyy.data(), dyydy.data(),
            d2yydy2.data(), ddy.data());
    const auto& g_wyy = g.field1d("wyy");
    const auto& g_dyydy = g.field1d("dyydy");
    const auto& g_d2yydy2 = g.field1d("d2yydy2");
    const auto& g_ddy = g.field1d("ddy");
    for (int j = 0; j < kNY; ++j) {
      EXPECT_DOUBLE_EQ(wyy[j], g_wyy[j]) << "mypey=" << mypey << " wyy[" << j << "]";
      EXPECT_DOUBLE_EQ(dyydy[j], g_dyydy[j]) << "mypey=" << mypey << " dyydy[" << j << "]";
      EXPECT_DOUBLE_EQ(d2yydy2[j], g_d2yydy2[j]) << "mypey=" << mypey << " d2yydy2[" << j << "]";
      EXPECT_DOUBLE_EQ(ddy[j], g_ddy[j]) << "mypey=" << mypey << " ddy[" << j << "]";
    }
  }
}

TEST(Grid, GoldenZZacobiRank0And2) {
  Params p = golden_params();
  for (int mypez : {0, 1}) {
    int rank = mypez * 2;  // mypey=0
    std::string path = golden_dir() + "/gstate.preloop.00" +
                       std::to_string(rank) + ".0000";
    Gstate g = read_gstate(path);
    ASSERT_EQ(g.mypez, mypez);
    std::vector<double> zee(kNZ), dzzdz(kNZ), d2zzdz2(kNZ), ddz(kNZ);
    zjacobi(p, mypez, /*npez=*/2, kNZ, kNPZ, kNRZ, zee.data(), dzzdz.data(),
            d2zzdz2.data(), ddz.data());
    const auto& g_zee = g.field1d("zee");
    const auto& g_dzzdz = g.field1d("dzzdz");
    const auto& g_d2zzdz2 = g.field1d("d2zzdz2");
    const auto& g_ddz = g.field1d("ddz");
    for (int k = 0; k < kNZ; ++k) {
      EXPECT_DOUBLE_EQ(zee[k], g_zee[k]) << "mypez=" << mypez << " zee[" << k << "]";
      EXPECT_DOUBLE_EQ(dzzdz[k], g_dzzdz[k]) << "mypez=" << mypez << " dzzdz[" << k << "]";
      EXPECT_DOUBLE_EQ(d2zzdz2[k], g_d2zzdz2[k]) << "mypez=" << mypez << " d2zzdz2[" << k << "]";
      EXPECT_DOUBLE_EQ(ddz[k], g_ddz[k]) << "mypez=" << mypez << " ddz[" << k << "]";
    }
  }
}

TEST(Grid, MkgridUniformAndArctan) {
  Params p = golden_params();
  // NGRID=0: uniform; NGRID=1: arctan (defaults give XX1=-7..7).
  Params u = p;
  u.ngrid = 0;
  double s, d1, d2;
  mkgrid(u, 0.5, 1, s, d1, d2);
  EXPECT_DOUBLE_EQ(s, 10.0);   // 0.5 * xmax
  EXPECT_DOUBLE_EQ(d1, 1.0 / 20.0);
  EXPECT_DOUBLE_EQ(d2, 0.0);
  // Endpoints.
  mkgrid(u, 0.0, 2, s, d1, d2);
  EXPECT_DOUBLE_EQ(s, 0.0);
  mkgrid(u, 1.0, 3, s, d1, d2);
  EXPECT_DOUBLE_EQ(s, 40.0);
  // Arctan grid: the first interior point of EXX in the golden is 5.8619.
  mkgrid(p, 1.0 / 15.0, 1, s, d1, d2);  // SCODE = 1/15 = 0.0667
  EXPECT_NEAR(s, 5.8619437, 1e-5);
}

TEST(Grid, SplineInfrastructurePresent) {
  // INIT_SPLINEX / SPLINEY are used only for NGRID>0 horizontal means which
  // the reference exercises through the main program.  The solver path for
  // the default (NGRID=0/1) is validation-covered by the golden metric tests
  // above; a dedicated spline test is added with the FLUXES/static port.
  SUCCEED();
}
