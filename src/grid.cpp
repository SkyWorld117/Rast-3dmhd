// grid.cpp - port of MKGRID / XJACOBI / YJACOBI / ZJACOBI.
//
// Expression structure of the original Fortran is preserved line-for-line so
// the generated values match the golden reference bit-for-bit.
#include "grid.hpp"

#include <cmath>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "fields.hpp"
#include "numcompat.hpp"

namespace r3d {

namespace {
double smax_for(const Params& p, int icoord) {
  switch (icoord) {
    case 1: return p.xmax;
    case 2: return p.ymax;
    case 3: return p.zmax;
    default:
      fprintf(stderr, "MKGRID: invalid ICOORD %d\n", icoord);
      exit(EXIT_FAILURE);
  }
}
}  // namespace

void mkgrid(const Params& p, double scode, int icoord, double& sphys,
            double& dsds, double& d2sds2) {
  const double onepi = 2.0 * asin(1.0);

  if (p.ngrid == 0) {
    double smax = smax_for(p, icoord);
    sphys = scode * smax;
    dsds = 1.0 / smax;
    d2sds2 = 0.0;
    return;
  }

  if (p.ngrid == 1) {
    double s1, s2, smax = 0;
    switch (icoord) {
      case 1: s1 = p.xx1; s2 = p.xx2; smax = p.xmax; break;
      case 2: s1 = p.yy1; s2 = p.yy2; smax = p.ymax; break;
      case 3: s1 = p.zz1; s2 = p.zz2; smax = p.zmax; break;
      default:
        fprintf(stderr, "MKGRID: invalid ICOORD %d\n", icoord);
        exit(EXIT_FAILURE);
    }
    double a1 = (s2 - s1) / smax;
    double a3 = atan(s1);
    double a2 = atan(s2) - a3;
    if (scode == 0.0) {
      sphys = 0.0;
    } else if (scode == 1.0) {
      sphys = smax;
    } else {
      sphys = (tan(a2 * scode + a3) - s1) / a1;
    }
    dsds = a1 / a2 / (1.0 + (s1 + a1 * sphys) * (s1 + a1 * sphys));
    double t = (s1 + a1 * sphys);
    // Fortran: -2*A1**2*T/A2/(1+T**2)**2  -> divide by the square of (1+T^2)
    double denom = (1.0 + t * t);
    d2sds2 = -2.0 * a1 * a1 * t / a2 / (denom * denom);
    return;
  }

  if (p.ngrid == 2) {
    double a = 0, b = 0, c = 0, dpar = 0, smax = 0;
    switch (icoord) {
      case 1: a = p.xa; b = p.xb; c = p.xc; dpar = p.xd; smax = p.xmax; break;
      case 2: a = p.ya; b = p.yb; c = p.yc; dpar = p.yd; smax = p.ymax; break;
      case 3: a = p.za; b = p.zb; c = p.zc; dpar = p.zd; smax = p.zmax; break;
      default:
        fprintf(stderr, "MKGRID: invalid ICOORD %d\n", icoord);
        exit(EXIT_FAILURE);
    }
    if (c <= 1.0e-9) {
      sphys = scode * smax;
      dsds = 1.0 / smax;
      d2sds2 = 0.0;
      return;
    }
    double sh = (1.0 - dpar) * onepi /
                (dpar * atan((a - b) * c) + 2.0 * atan(b * c) -
                 dpar * atan((a + b) * c));
    double sk = 2.0 * c * onepi * smax /
                (2.0 * c * onepi + sh *
                     (2.0 * c *
                          (atan((a - b) * c) * (a - b) -
                           atan((a + b) * c) * (a + b) +
                           atan((a - b - 1.0) * c) +
                           atan((a + b - 1.0) * c) * (a + b - 1.0) +
                           atan((1.0 - a + b) * c) * (a - b)) -
                      log(1.0 + (a - b) * (a - b) * c * c) +
                      log(1.0 + (a + b) * (a + b) * c * c) -
                      log(1.0 + (a + b - 1.0) * (a + b - 1.0) * c * c) +
                      log(1.0 + (1.0 - a + b) * (1.0 - a + b) * c * c)));
    dsds = 1.0 / (sk * (1.0 + sh / onepi *
                            (atan(c * (scode - a - b)) +
                             atan(c * (a - scode - b)))));
    d2sds2 = -dsds * dsds * dsds * c * sh * sk / onepi *
             (1.0 / (1.0 + c * c * (a + b - scode) * (a + b - scode)) -
              1.0 / (1.0 + c * c * (a - b - scode) * (a - b - scode)));
    sphys = sk / (2.0 * c * onepi) *
            (2.0 * c * onepi * scode +
             sh * (2.0 * c *
                       (atan((a - b) * c) * (a - b) -
                        atan((a + b) * c) * (a + b) +
                        atan((a - b - scode) * c) * scode +
                        atan((a + b - scode) * c) * (a + b - scode) +
                        atan((scode - a + b) * c) * (a - b)) -
                   log(1.0 + (a - b) * (a - b) * c * c) +
                   log(1.0 + (a + b) * (a + b) * c * c) -
                   log(1.0 + (a + b - scode) * (a + b - scode) * c * c) +
                   log(1.0 + (scode - a + b) * (scode - a + b) * c * c)));
    if (scode == 0.0) sphys = 0.0;
    if (scode == 1.0) sphys = smax;
    return;
  }

  fprintf(stderr, "MKGRID: invalid NGRID %d\n", p.ngrid);
  exit(EXIT_FAILURE);
}

void xjacobi(const Params& p, int nx, double* exx, double* dxxdx,
             double* d2xxdx2, double* ddx) {
  // Computational grid: evenly spaced x values between zero and one.
  // ORX = 1.0E00/FLOAT(NX-IX-1) computes in single precision.
  double orx = f4(1.0f / (float)(nx - p.ix - 1));
  for (int i = 0; i < nx - p.ix - 1; ++i)      // Fortran I=0..NX-IX-1
    exx[i + 1] = i * orx;
  exx[1] = 0.0;
  exx[nx - p.ix] = 1.0;                        // EXX(NX-IX+1)

  for (int i = 1; i <= nx - p.ix; ++i) {       // Fortran I=2..NX-IX+1
    double scode = exx[i];
    double sphys, dsds, d2sds2;
    mkgrid(p, scode, 1, sphys, dsds, d2sds2);
    exx[i] = sphys;
    dxxdx[i] = dsds;
    d2xxdx2[i] = d2sds2;
    ddx[i] = orx * (1.0 / dsds);
  }
}

void yjacobi(const Params& p, int mypey, int npey, int ny, int npy, int nry,
             double* wyy, double* dyydy, double* d2yydy2, double* ddy) {
  const int iy = p.iy;
  const int iyh = iy / 2;

  // Full computational grid: evenly spaced y values between zero and one.
  // ORY = 1.0E00/FLOAT(NPY-1) computes in single precision.
  double ory = f4(1.0f / (float)(npy - 1));
  std::vector<double> yy(npy);
  for (int j = 0; j < npy; ++j) yy[j] = j * ory;
  yy[0] = 0.0;
  yy[npy - 1] = 1.0;

  if (mypey == 0) {
    for (int j = 0; j < iyh; ++j) {   // Fortran J=1..IY/2
      double sphys, dsds, d2sds2;
      mkgrid(p, 0.0, 2, sphys, dsds, d2sds2);
      wyy[j] = sphys; dyydy[j] = dsds; d2yydy2[j] = d2sds2;
    }
    for (int j = iyh; j < nry + iy; ++j) {  // Fortran J=IY/2+1..NRY+IY
      double sphys, dsds, d2sds2;
      mkgrid(p, yy[j - iyh], 2, sphys, dsds, d2sds2);
      wyy[j] = sphys; dyydy[j] = dsds; d2yydy2[j] = d2sds2;
    }
  } else if (mypey == npey - 1) {
    for (int j = 0; j < nry + iyh; ++j) {  // Fortran J=1..NRY+IY/2
      double sphys, dsds, d2sds2;
      mkgrid(p, yy[j + npy - nry - iyh], 2, sphys, dsds, d2sds2);
      wyy[j] = sphys; dyydy[j] = dsds; d2yydy2[j] = d2sds2;
    }
    for (int j = nry + iyh; j < nry + iy; ++j) {  // J=NRY+IY/2+1..NRY+IY
      double sphys, dsds, d2sds2;
      mkgrid(p, 1.0, 2, sphys, dsds, d2sds2);
      wyy[j] = sphys; dyydy[j] = dsds; d2yydy2[j] = d2sds2;
    }
  } else {
    for (int j = 0; j < ny; ++j) {  // Fortran J=1..NY
      double sphys, dsds, d2sds2;
      mkgrid(p, yy[j + mypey * nry - iyh], 2, sphys, dsds, d2sds2);
      wyy[j] = sphys; dyydy[j] = dsds; d2yydy2[j] = d2sds2;
    }
  }
  for (int j = 0; j < ny; ++j) ddy[j] = ory * (1.0 / dyydy[j]);
}

void zjacobi(const Params& p, int mypez, int npez, int nz, int npz, int nrz,
             double* zee, double* dzzdz, double* d2zzdz2, double* ddz) {
  const int ilap = p.ilap;
  const int ilaph = ilap / 2;

  // ORZ = 1.0E00/FLOAT(NPZ-1) computes in single precision.
  double orz = f4(1.0f / (float)(npz - 1));
  std::vector<double> zz(npz);
  for (int k = 0; k < npz; ++k) zz[k] = k * orz;
  zz[0] = 0.0;
  zz[npz - 1] = 1.0;

  if (mypez == 0) {
    for (int k = 0; k < ilaph; ++k) {   // Fortran K=1..ILAP/2
      double sphys, dsds, d2sds2;
      mkgrid(p, 0.0, 3, sphys, dsds, d2sds2);
      zee[k] = sphys; dzzdz[k] = dsds; d2zzdz2[k] = d2sds2;
    }
    for (int k = ilaph; k < nrz + ilap; ++k) {  // K=ILAP/2+1..NRZ+ILAP
      double sphys, dsds, d2sds2;
      mkgrid(p, zz[k - ilaph], 3, sphys, dsds, d2sds2);
      zee[k] = sphys; dzzdz[k] = dsds; d2zzdz2[k] = d2sds2;
    }
  } else if (mypez == npez - 1) {
    for (int k = 0; k < nrz + ilaph; ++k) {  // K=1..NRZ+ILAP/2
      double sphys, dsds, d2sds2;
      mkgrid(p, zz[k + npz - nrz - ilaph], 3, sphys, dsds, d2sds2);
      zee[k] = sphys; dzzdz[k] = dsds; d2zzdz2[k] = d2sds2;
    }
    for (int k = nrz + ilaph; k < nrz + ilap; ++k) {  // K=NRZ+ILAP/2+1..NRZ+ILAP
      double sphys, dsds, d2sds2;
      mkgrid(p, 1.0, 3, sphys, dsds, d2sds2);
      zee[k] = sphys; dzzdz[k] = dsds; d2zzdz2[k] = d2sds2;
    }
  } else {
    for (int k = 0; k < nz; ++k) {  // K=1..NZ
      double sphys, dsds, d2sds2;
      mkgrid(p, zz[k + mypez * nrz - ilaph], 3, sphys, dsds, d2sds2);
      zee[k] = sphys; dzzdz[k] = dsds; d2zzdz2[k] = d2sds2;
    }
  }
  for (int k = 0; k < nz; ++k) ddz[k] = orz * (1.0 / dzzdz[k]);
}

void build_grid_metrics(const Params& p, const Topology& t, SimState& s) {
  xjacobi(p, s.d.nx, s.exx.data(), s.dxxdx.data(), s.d2xxdx2.data(), s.ddx.data());
  yjacobi(p, t.mypey, t.npey, s.d.ny, p.npy, s.d.nry, s.wyy.data(),
          s.dyydy.data(), s.d2yydy2.data(), s.ddy.data());
  zjacobi(p, t.mypez, t.npez, s.d.nz, p.npz, s.d.nrz, s.zee.data(),
          s.dzzdz.data(), s.d2zzdz2.data(), s.ddz.data());
}

}  // namespace r3d
