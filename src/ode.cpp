// ode.cpp - Bulirsch-Stoer machinery and the static-profile construction,
// ported from BSSTEP / MMID / RZEXTR / DERIVS in 3dmhdsub.f.
#include "ode.hpp"

#include <cmath>
#include <cstdlib>

#include "grid.hpp"
#include "numcompat.hpp"
#include <cstdio>

namespace r3d {

namespace {
constexpr int kNMax = 10;   // max state size
constexpr int kIMax = 11;   // max extrapolation stages
constexpr int kNuse = 7;    // stages used
constexpr double kShrink = 0.95;
constexpr double kGrow = 1.2;
const int kNsEq[kIMax] = {2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96};
}  // namespace

void derivs_static(const Params& p, double zz, const double* y, double* dydx) {
  // MKGRID(ZZ,Z,DZZDZ,D2ZZDZ2,3)
  double z, dzzdz, d2zzdz2;
  mkgrid(p, zz, 3, z, dzzdz, d2zzdz2);
  // RKAPST = (POLYS+1)*THETA/GRAV, as computed in main before STATIC runs.
  double rkapst = (p.grav == 0.0) ? 1.0 : (p.polys + 1.0) * p.theta / p.grav;
  double rkap;
  if (p.pzp == 0.0) {
    rkap = 1.0;
  } else {
    rkap = 1.0 + (rkapst - 1.0) / 2.0 * (1.0 + tanh((z - p.pzp) / p.sigma));
  }
  dydx[0] = p.theta / rkap / dzzdz;             // DYDX(1)=THETA/RKAP/DZZDZ
  dydx[1] = p.grav / (1.0 + y[0]) / dzzdz;      // DYDX(2)=GRAV/(1+Y(1))/DZZDZ
}

// RZEXTR (NR p. 566): diagonal rational-function extrapolation.
void rzextr(int iest, double xest, const double* yest, double* yz,
            double d[kNMax][kNuse], double* dy, int nv, double* xbuf) {
  xbuf[iest - 1] = xest;
  if (iest == 1) {
    for (int j = 0; j < nv; ++j) {
      yz[j] = yest[j];
      d[j][0] = yest[j];
      dy[j] = yest[j];
    }
  } else {
    int m1 = iest < kNuse ? iest : kNuse;
    double fx[kNuse + 1];
    // Fortran FX(K+1) = X(IEST-K) / XEST   (X is 1-based)
    for (int k = 0; k < m1 - 1; ++k) fx[k + 1] = xbuf[iest - k - 2] / xest;
    for (int j = 0; j < nv; ++j) {
      double yy = yest[j];
      double v = d[j][0];
      double c = yy;
      d[j][0] = yy;
      double ddy;
      for (int k = 1; k < m1; ++k) {  // Fortran K=2..M1 -> 0-based 1..m1-1
        double b1 = fx[k] * v;
        double b = b1 - c;
        if (b != 0.0) {
          b = (c - v) / b;
          ddy = c * b;
          c = b1 * b;
        } else {
          ddy = v;
        }
        v = d[j][k];
        d[j][k] = ddy;
        yy += ddy;
      }
      dy[j] = ddy;
      yz[j] = yy;
    }
  }
}

// MMID (NR p. 560): modified midpoint rule.
void mmid(const Params& p, const double* y, const double* dydx, int nvar,
          double xs, double htot, int nstp, double* yout, DerivFn derivs) {
  double h = htot / nstp;
  double ym[kNMax], yn[kNMax];
  for (int i = 0; i < nvar; ++i) {
    ym[i] = y[i];
    yn[i] = y[i] + h * dydx[i];
  }
  double x = xs + h;
  double ytemp[kNMax];
  derivs(p, x, yn, ytemp);  // YOUT holds dydx here
  double h2 = 2.0 * h;
  for (int n = 1; n < nstp; ++n) {
    for (int i = 0; i < nvar; ++i) {
      double swap = ym[i] + h2 * ytemp[i];
      ym[i] = yn[i];
      yn[i] = swap;
    }
    x += h;
    derivs(p, x, yn, ytemp);
  }
  for (int i = 0; i < nvar; ++i)
    yout[i] = 0.5 * (ym[i] + yn[i] + h * ytemp[i]);
}

void bsstep(const Params& p, double* y, double* dydx, int nv, double& x,
            double htry, double eps, double& hdid, double& hnext,
            DerivFn derivs, double* xbuf) {
  double h = htry;
  double xsav = x;
  double ysav[kNMax], dysav[kNMax];
  for (int i = 0; i < nv; ++i) {
    ysav[i] = y[i];
    dysav[i] = dydx[i];
  }
  double yseq[kNMax], yerr[kNMax];
  double d[kNMax][kNuse];

  for (;;) {  // Fortran line 1: on failure, shrink and retry
    for (int i = 1; i <= kIMax; ++i) {
      mmid(p, ysav, dysav, nv, xsav, h, kNsEq[i - 1], yseq, derivs);
      double xest = (h / kNsEq[i - 1]) * (h / kNsEq[i - 1]);
      rzextr(i, xest, yseq, y, d, yerr, nv, xbuf);
      if (getenv("ODE_DEBUG")) fprintf(stderr, "i=%d y=(%.6g %.6g) yerr=(%.3g %.3g)\n", i, y[0], y[1], yerr[0], yerr[1]);
      double errmax = 0.0;
      for (int j = 0; j < nv; ++j) {
        double e = std::fabs(yerr[j] / y[j]);
        if (e > errmax) errmax = e;
      }
      errmax /= eps;
      if (errmax < 1.0) {
        x += h;
        hdid = h;
        if (i == kNuse) {
          hnext = h * kShrink;
        } else if (i == kNuse - 1) {
          hnext = h * kGrow;
        } else {
          hnext = (h * kNsEq[kNuse - 1]) / kNsEq[i - 1];
        }
        return;
      }
    }
    h = 0.25 * h / std::pow(2.0, double((kIMax - kNuse) / 2));
    if (x + h == x) {
      // BSSTEP: step-size underflow -> terminate.
      __builtin_trap();
    }
  }
}

void static_profile(const Params& p, int npz, double* tprof, double* rprof) {
  // DZZ = 1.0E00/FLOAT(NPZ-1) : single-precision division, then widened.
  double dzz = f4(1.0f / (float)(npz - 1));
  const int nv = 2;
  double y[nv] = {0.0, 0.0};  // FFZ, PLN
  tprof[0] = 1.0;
  rprof[0] = 1.0;
  double zz = 0.0;
  double xbuf[kIMax];
  for (int k = 1; k < npz; ++k) {
    double dydx[nv];
    derivs_static(p, zz, y, dydx);
    double htry = dzz;
    double hdid, hnext;
    bsstep(p, y, dydx, nv, zz, htry, /*eps=*/1.0e-12, hdid, hnext, derivs_static,
           xbuf);
    // STATIC requires HDID == HTRY; the original terminates otherwise.
    if (hdid != htry) {
      __builtin_trap();
    }
    tprof[k] = 1.0 + y[0];
    rprof[k] = std::exp(y[1]) / (1.0 + y[0]);
  }
}

}  // namespace r3d
