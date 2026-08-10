// bcon.cpp - boundary conditions (port of 3dmhdsub.f BCON).
#include "bcon.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "numcompat.hpp"

namespace r3d {

void bcon(const Params& p, const Derived& d, const Topology& t, SimState& s) {
  if (p.lshr) {
    fprintf(stderr, "rast-3dmhd: BCON: LSHR boundary conditions not ported\n");
    exit(EXIT_FAILURE);
  }
  const int nx = d.nx, ny = d.ny, nz = d.nz;
  const int i1 = 1, i2 = nx - p.ix;
  const int j1 = 1, j2 = ny - p.iy;
  const int k1 = p.ilap / 2, k2 = nz - 1 - p.ilap / 2;
  const double c13 = d.c13, c23 = d.c23, c43 = d.c43;
  const double timt = s.timc + s.timi;   // TIMT = TIMI + TIMC

  // ===================================================================
  // Top (upper boundary; the first z rank).
  // ===================================================================
  if (t.mypez == 0) {
    // Stress-free, impenetrable: reflect the horizontal momenta of the
    // first interior plane linearly through zero.
    for (int j = j1; j <= j2; ++j)
      for (int i = i1; i <= i2; ++i)
        s.ru.at(i, j, k1) =
            s.ro.at(i, j, k1) *
            (c43 * s.ru.at(i, j, k1 + 1) / s.ro.at(i, j, k1 + 1) -
             c13 * s.ru.at(i, j, k1 + 2) / s.ro.at(i, j, k1 + 2));
    for (int j = j1; j <= j2; ++j)
      for (int i = i1; i <= i2; ++i)
        s.rv.at(i, j, k1) =
            s.ro.at(i, j, k1) *
            (c43 * s.rv.at(i, j, k1 + 1) / s.ro.at(i, j, k1 + 1) -
             c13 * s.rv.at(i, j, k1 + 2) / s.ro.at(i, j, k1 + 2));
    for (int j = j1; j <= j2; ++j)
      for (int i = i1; i <= i2; ++i)
        s.rw.at(i, j, k1) = 0.0;

    if ((p.tp != 0.0) && (p.itcon == 0 || p.itcon == 1)) {
      // Plume temperature / flux condition held on the upper boundary.
      double tpr = (p.tc != 0.0) ? p.tp + (1.0 - p.tp) * exp(-timt / p.tc) : p.tp;
      // CLN = -4*LOG(2)/HH^2 with single-precision LOG(2).
      const double cln = f4(-4.0f * logf(2.0f)) / (p.hh * p.hh);
      if (p.itcon == 0) {
        // Gaussian temperature profile (constant temperature, perturbed).
        for (int j = j1; j <= j2; ++j)
          for (int i = i1; i <= i2; ++i)
            s.tt.at(i, j, k1) =
                1.0 - (1.0 - tpr) * exp(cln * (s.exx[i] - p.xp) * (s.exx[i] - p.xp)) /
                          p.hh *
                          exp(cln * (s.wyy[j] - p.yp) * (s.wyy[j] - p.yp)) / p.hh;
      } else {
        // Gaussian flux profile.
        // DZ = 1.0E00/FLOAT(NPZ-1)/DZZDZ : 1/(NPZ-1) is single precision.
        const double dz = f4(1.0f / (float)(p.npz - 1)) / s.dzzdz[k1];
        for (int j = j1; j <= j2; ++j)
          for (int i = i1; i <= i2; ++i)
            s.tt.at(i, j, k1) =
                c43 * s.tt.at(i, j, k1 + 1) - c13 * s.tt.at(i, j, k1 + 2) -
                c23 * dz *
                    (p.theta -
                     (p.theta - tpr) *
                         exp(cln * ((s.exx[i] - p.xp) * (s.exx[i] - p.xp))) / p.hh *
                         exp(cln * ((s.wyy[j] - p.yp) * (s.wyy[j] - p.yp))) / p.hh);
      }
    } else {
      if (p.itcon == 0 || p.itcon == 2 || p.itcon == 4) {
        // Constant temperature upper boundary (with optional half-domain
        // nonuniformity for ITCON=4).
        for (int j = j1; j <= j2; ++j)
          for (int i = i1; i <= i2; ++i) {
            if (p.lrem) {
              s.tt.at(i, j, k1) = s.tb;  // TU placeholder = TB for LREM
            } else {
              s.tt.at(i, j, k1) = 1.0;
              if (p.itcon == 4 && i <= nx / 2) s.tt.at(i, j, k1) = 1.0 - p.tp;
            }
          }
      }
      if (p.itcon == 1) {
        // Constant flux upper boundary, no plume perturbation.
        const double dz = f4(1.0f / (float)(p.npz - 1)) / s.dzzdz[k1];
        for (int j = j1; j <= j2; ++j)
          for (int i = i1; i <= i2; ++i) {
            // (LREM branch would add DZTU here; not ported.)
            s.tt.at(i, j, k1) = c43 * s.tt.at(i, j, k1 + 1) -
                                c13 * s.tt.at(i, j, k1 + 2) - c23 * dz * p.theta;
          }
      }
    }
    if (p.lmag && p.ibcon == 0) {
      for (int j = j1; j <= j2; ++j)
        for (int i = i1; i <= i2; ++i)
          s.bz.at(i, j, k1) = 0.0;
    }
  }

  // ===================================================================
  // Bottom (lower boundary; the last z rank).
  // ===================================================================
  if (t.mypez == t.npez - 1) {
    for (int j = j1; j <= j2; ++j)
      for (int i = i1; i <= i2; ++i)
        s.ru.at(i, j, k2) =
            s.ro.at(i, j, k2) *
            (c43 * s.ru.at(i, j, k2 - 1) / s.ro.at(i, j, k2 - 1) -
             c13 * s.ru.at(i, j, k2 - 2) / s.ro.at(i, j, k2 - 2));
    for (int j = j1; j <= j2; ++j)
      for (int i = i1; i <= i2; ++i)
        s.rv.at(i, j, k2) =
            s.ro.at(i, j, k2) *
            (c43 * s.rv.at(i, j, k2 - 1) / s.ro.at(i, j, k2 - 1) -
             c13 * s.rv.at(i, j, k2 - 2) / s.ro.at(i, j, k2 - 2));
    for (int j = j1; j <= j2; ++j)
      for (int i = i1; i <= i2; ++i)
        s.rw.at(i, j, k2) = 0.0;

    if (p.izcon == 0) {
      // Constant temperature lower boundary.
      for (int j = j1; j <= j2; ++j)
        for (int i = i1; i <= i2; ++i)
          s.tt.at(i, j, k2) = s.tb;
    } else if (p.izcon == 1) {
      // Constant flux lower boundary.
      const double dz = f4(1.0f / (float)(p.npz - 1)) / s.dzzdz[k2];
      for (int j = j1; j <= j2; ++j)
        for (int i = i1; i <= i2; ++i) {
          // (LREM branch would add DZTB here; not ported.)
          s.tt.at(i, j, k2) = c43 * s.tt.at(i, j, k2 - 1) -
                              c13 * s.tt.at(i, j, k2 - 2) +
                              c23 * dz * p.theta / s.rkapa[k2];
        }
    } else {
      fprintf(stderr, "rast-3dmhd: BCON: invalid IZCON %d\n", p.izcon);
      exit(EXIT_FAILURE);
    }
    if (p.lmag && p.ibcon == 0) {
      for (int j = j1; j <= j2; ++j)
        for (int i = i1; i <= i2; ++i)
          s.bz.at(i, j, k2) = 0.0;
    }
  }
}

}  // namespace r3d
