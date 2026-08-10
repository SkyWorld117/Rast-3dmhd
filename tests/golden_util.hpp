// golden_util.hpp - shared helpers for the golden-matching tests: build the
// tiny golden geometry, load a golden gstate into a SimState, and compare
// C++ fields against golden arrays (interiors only - ghost memory in the
// original Fortran is uninitialised).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "fields.hpp"
#include "gstate_reader.hpp"
#include "params.hpp"
#include "topology.hpp"

namespace r3dtest {

// The golden geometry: NPX=16, NPY=16, NPZ=32, NPEY=NPEZ=2.
inline r3d::Params golden_params() {
  r3d::Params p;
  p.npx = 16; p.npy = 16; p.npz = 32;
  p.npey = 2;
  return p;
}

inline std::string golden_dir() {
  const char* e = getenv("RAST_3DMHD_GOLDEN");
  return e ? std::string(e) : std::string("golden/ref");
}

// Preset rank coordinates for the golden 2x2 grid (no MPI needed).
inline r3d::Topology golden_topo(int mype) {
  r3d::Topology t;
  t.npe = 4;
  t.npey = 2;
  t.npez = 2;
  t.mype = mype;
  r3d::RankXY r = r3d::rank_to_xy(mype, t.npey);
  t.mypey = r.mypey;
  t.mypez = r.mypez;
  return t;
}

// Copy the golden 3D/1D arrays (from a gstate file) into a SimState.
inline void load_state(r3d::SimState& s, const Gstate& g) {
  const std::vector<double>* src[27] = {
      &g.f3d[Gstate::idx3d("ru")], &g.f3d[Gstate::idx3d("rv")],
      &g.f3d[Gstate::idx3d("rw")], &g.f3d[Gstate::idx3d("ro")],
      &g.f3d[Gstate::idx3d("tt")], &g.f3d[Gstate::idx3d("uu")],
      &g.f3d[Gstate::idx3d("vv")], &g.f3d[Gstate::idx3d("ww")],
      &g.f3d[Gstate::idx3d("fu")], &g.f3d[Gstate::idx3d("fv")],
      &g.f3d[Gstate::idx3d("fw")], &g.f3d[Gstate::idx3d("fr")],
      &g.f3d[Gstate::idx3d("ft")], &g.f3d[Gstate::idx3d("zru")],
      &g.f3d[Gstate::idx3d("zrv")], &g.f3d[Gstate::idx3d("zrw")],
      &g.f3d[Gstate::idx3d("zro")], &g.f3d[Gstate::idx3d("ztt")],
      &g.f3d[Gstate::idx3d("ww1")], &g.f3d[Gstate::idx3d("ww2")],
      &g.f3d[Gstate::idx3d("ww3")], &g.f3d[Gstate::idx3d("bx")],
      &g.f3d[Gstate::idx3d("by")], &g.f3d[Gstate::idx3d("bz")],
      &g.f3d[Gstate::idx3d("zbx")], &g.f3d[Gstate::idx3d("zby")],
      &g.f3d[Gstate::idx3d("zbz")]};
  r3d::Field* dst[27] = {&s.ru,  &s.rv,  &s.rw,  &s.ro,  &s.tt,
                         &s.uu,  &s.vv,  &s.ww,  &s.fu,  &s.fv,
                         &s.fw,  &s.fr,  &s.ft,  &s.zru, &s.zrv,
                         &s.zrw, &s.zro, &s.ztt, &s.ww1, &s.ww2,
                         &s.ww3, &s.bx,  &s.by,  &s.bz,  &s.zbx,
                         &s.zby, &s.zbz};
  for (int i = 0; i < 27; ++i) {
    r3d::Field tmp(r3d::Grid3{g.nx, g.ny, g.nz}, src[i]->data());
    dst[i]->copy_from(tmp);
  }
  struct Q { const char* name; std::vector<double>* dst; };
  Q q[14] = {
      {"exx", &s.exx}, {"dxxdx", &s.dxxdx}, {"d2xxdx2", &s.d2xxdx2},
      {"ddx", &s.ddx}, {"wyy", &s.wyy}, {"dyydy", &s.dyydy},
      {"d2yydy2", &s.d2yydy2}, {"ddy", &s.ddy}, {"zee", &s.zee},
      {"dzzdz", &s.dzzdz}, {"d2zzdz2", &s.d2zzdz2}, {"ddz", &s.ddz},
      {"rkapa", &s.rkapa}, {"dkapa", &s.dkapa}};
  for (auto& e : q)
    e.dst->assign(g.f1d[Gstate::idx1d(e.name)].begin(),
                  g.f1d[Gstate::idx1d(e.name)].end());
  s.dt = g.scalars[0];
  s.timc = g.scalars[1];
  s.timt = g.scalars[2];
  s.timi = g.scalars[3];
  s.umach = g.scalars[4];
  s.nit = g.nit;
}

// Max abs difference between a C++ field and a golden array over the
// interior cells only (i in [1,nx-ix], j in [1,ny-iy], k in [ilap/2,nz-1-ilap/2]).
inline double max_err_interior(const r3d::Field& f, const std::vector<double>& g,
                               int ix, int iy, int ilap) {
  const int nx = f.nx(), ny = f.ny(), nz = f.nz();
  double m = 0.0;
  for (int k = ilap / 2; k <= nz - 1 - ilap / 2; ++k)
    for (int j = 1; j <= ny - iy; ++j)
      for (int i = 1; i <= nx - ix; ++i)
        m = std::max(m, std::fabs(
                            f.at(i, j, k) -
                            g[(long)k * nx * ny + (long)j * nx + i]));
  return m;
}

}  // namespace r3dtest
