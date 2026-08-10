// fluxes.cpp - the FLUXES kernel (port of 3dmhdsub.f FLUXES).
//
// Index map (Fortran 1-based -> C++ 0-based):
//   interior x : I=2..NX-IX+1      -> i in [1, nx-ix]
//   interior y : J=2..NY-IY+1      -> j in [1, ny-iy]
//   interior z : K=ILAP/2+1..NZ-ILAP/2 -> k in [ilap/2, nz-1-ilap/2]
// All array writes/reads in this file use the flat x-fastest indexing of
// the original column-major arrays, so expressions map 1:1.
#include "fluxes.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "communicate.hpp"

// Parallel sweeps with OpenMP (CPU backend only).  Every sweep below is a
// stencil writing only its own cell, and there are no reductions in this file,
// so threading the k axis preserves the golden bit-exact result.  Compiled out
// when _OPENMP is undefined (the GPU backend builds these files as host code
// without -fopenmp, where a thread pool would be wasted per rank).
#ifdef _OPENMP
#define R3D_OMP_FOR _Pragma("omp parallel for schedule(static)")
#else
#define R3D_OMP_FOR
#endif

namespace r3d {

    // Optional advanced features of the original (LREM stratification / flux
    // relaxation / embedded heat loss / shear) are ported as CPU paths; the GPU
    // backend targets the default configuration.  The default configuration
    // (LREM=false, RLAX=0, ITC!=2, LSHR=false, ID=0) is exercised by the golden
    // tests and must remain bit-exact.
    namespace {

        [[noreturn]] void unsupported(const char* what) {
            fprintf(stderr, "rast-3dmhd: FLUXES: unsupported feature enabled: %s\n", what);
            exit(EXIT_FAILURE);
        }

    }  // namespace

    void fluxes(const Params& p, const Derived& d, const Topology& t, const double rkapa[],
                const double dkapa[], SimState& s) {
        if (p.lshr) unsupported("LSHR");
        if (p.lrem) unsupported("LREM");
        if (p.rlax > 0.0) unsupported("flux relaxation (RLAX)");
        if (p.id != 0) unsupported("density diffusion ID");
        if (p.itcon == 2 || p.itcon == 3) unsupported("ITCON embedded heat/thermal");
        if (p.ixcon != 0 || p.iycon != 0) unsupported("non-periodic horizontal boundaries");

        Field& RU  = s.ru;
        Field& RV  = s.rv;
        Field& RW  = s.rw;
        Field& RO  = s.ro;
        Field& TT  = s.tt;
        Field& UU  = s.uu;
        Field& VV  = s.vv;
        Field& WW  = s.ww;
        Field& FU  = s.fu;
        Field& FV  = s.fv;
        Field& FW  = s.fw;
        Field& FR  = s.fr;
        Field& FT  = s.ft;
        Field& WW1 = s.ww1;
        Field& WW2 = s.ww2;
        Field& WW3 = s.ww3;
        Field& BX  = s.bx;
        Field& BY  = s.by;
        Field& BZ  = s.bz;

        const int nx = d.nx, ny = d.ny, nz = d.nz;
        const int i1 = 1, i2 = nx - p.ix;                     // inclusive interior x
        const int j1 = 1, j2 = ny - p.iy;                     // inclusive interior y
        const int k1 = p.ilap / 2, k2 = nz - 1 - p.ilap / 2;  // inclusive interior z

        const double c13 = d.c13, c23 = d.c23, c43 = d.c43;
        const double ore = d.ore;
        const double hx = d.hx, h2x = d.h2x;
        const double hy = d.hy, h2y = d.h2y;
        const double hz = d.hz, h2z = d.h2z;

        // ===================================================================
        // FR from the divergence of the momentum.
        // ===================================================================
        R3D_OMP_FOR
        for (int k = k1; k <= k2; ++k) {
            for (int j = j1; j <= j2; ++j) {
                double tmpy = hy * s.dyydy[j];
                for (int i = i1; i <= i2; ++i) {
                    FR.at(i, j, k) = (RU.at(i - 1, j, k) - RU.at(i + 1, j, k)) * hx * s.dxxdx[i];
                    FR.at(i, j, k) -= (RV.at(i, j + 1, k) - RV.at(i, j - 1, k)) * tmpy;
                }
            }
        }
        // WW1 = dRW/dz (with one-sided 2nd-order stencils at the two z edges of
        // the domain, handled per rank).
        R3D_OMP_FOR
        for (int k = k1; k <= k2; ++k) {
            double tmpz = hz * s.dzzdz[k];
            for (int j = j1; j <= j2; ++j)
                for (int i = i1; i <= i2; ++i)
                    WW1.at(i, j, k) = (RW.at(i, j, k + 1) - RW.at(i, j, k - 1)) * tmpz;
        }
        if (t.mypez == 0) {
            double tmpz = hz * s.dzzdz[k1];
            for (int j = j1; j <= j2; ++j)
                for (int i = i1; i <= i2; ++i)
                    WW1.at(i, j, k1) = (4.0 * RW.at(i, j, k1 + 1) - RW.at(i, j, k1 + 2)) * tmpz;
        }
        if (t.mypez == t.npez - 1) {
            double tmpz = hz * s.dzzdz[k2];
            for (int j = j1; j <= j2; ++j)
                for (int i = i1; i <= i2; ++i)
                    WW1.at(i, j, k2) = (RW.at(i, j, k2 - 2) - 4.0 * RW.at(i, j, k2 - 1)) * tmpz;
        }
        R3D_OMP_FOR
        for (int k = k1; k <= k2; ++k)
            for (int j = j1; j <= j2; ++j)
                for (int i = i1; i <= i2; ++i) FR.at(i, j, k) -= WW1.at(i, j, k);

        // ===================================================================
        // Buoyancy, pressure, 1/rho and velocities.
        // ===================================================================
        R3D_OMP_FOR
        for (int k = k1; k <= k2; ++k)
            for (int j = j1; j <= j2; ++j)
                for (int i = i1; i <= i2; ++i) FW.at(i, j, k) = p.grav * RO.at(i, j, k);

        // WW1 = P = RHO*T ; RO = 1/RHO over the FULL local array (ghosts too).
        R3D_OMP_FOR
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    WW1.at(i, j, k) = RO.at(i, j, k) * TT.at(i, j, k);
                    RO.at(i, j, k)  = 1.0 / RO.at(i, j, k);
                }

        // Pressure gradients.
        R3D_OMP_FOR
        for (int k = k1; k <= k2; ++k) {
            double tmpz = hz * s.dzzdz[k];
            for (int j = j1; j <= j2; ++j) {
                double tmpy = hy * s.dyydy[j];
                for (int i = i1; i <= i2; ++i) {
                    FW.at(i, j, k) -= (WW1.at(i, j, k + 1) - WW1.at(i, j, k - 1)) * tmpz;
                    FV.at(i, j, k) = (WW1.at(i, j - 1, k) - WW1.at(i, j + 1, k)) * tmpy;
                    FU.at(i, j, k) = (WW1.at(i - 1, j, k) - WW1.at(i + 1, j, k)) * hx * s.dxxdx[i];
                }
            }
        }

        // Velocities = momentum * 1/rho (full array).
        R3D_OMP_FOR
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) UU.at(i, j, k) = RU.at(i, j, k) * RO.at(i, j, k);
        R3D_OMP_FOR
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) VV.at(i, j, k) = RV.at(i, j, k) * RO.at(i, j, k);
        R3D_OMP_FOR
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) WW.at(i, j, k) = RW.at(i, j, k) * RO.at(i, j, k);

        // ===================================================================
        // Advection of momenta.
        // ===================================================================
        R3D_OMP_FOR
        for (int k = k1; k <= k2; ++k) {
            double tmpz = hz * s.dzzdz[k];
            for (int j = j1; j <= j2; ++j) {
                double tmpy = hy * s.dyydy[j];
                for (int i = i1; i <= i2; ++i) {
                    FU.at(i, j, k) -= (RU.at(i + 1, j, k) * UU.at(i + 1, j, k) -
                                       RU.at(i - 1, j, k) * UU.at(i - 1, j, k)) *
                                      hx * s.dxxdx[i];
                    FU.at(i, j, k) -= (RU.at(i, j + 1, k) * VV.at(i, j + 1, k) -
                                       RU.at(i, j - 1, k) * VV.at(i, j - 1, k)) *
                                      tmpy;
                    FU.at(i, j, k) -= (RU.at(i, j, k + 1) * WW.at(i, j, k + 1) -
                                       RU.at(i, j, k - 1) * WW.at(i, j, k - 1)) *
                                      tmpz;
                }
            }
        }
        R3D_OMP_FOR
        for (int k = k1; k <= k2; ++k) {
            double tmpz = hz * s.dzzdz[k];
            for (int j = j1; j <= j2; ++j) {
                double tmpy = hy * s.dyydy[j];
                for (int i = i1; i <= i2; ++i) {
                    FV.at(i, j, k) -= (RV.at(i + 1, j, k) * UU.at(i + 1, j, k) -
                                       RV.at(i - 1, j, k) * UU.at(i - 1, j, k)) *
                                      hx * s.dxxdx[i];
                    FV.at(i, j, k) -= (RV.at(i, j + 1, k) * VV.at(i, j + 1, k) -
                                       RV.at(i, j - 1, k) * VV.at(i, j - 1, k)) *
                                      tmpy;
                    FV.at(i, j, k) -= (RV.at(i, j, k + 1) * WW.at(i, j, k + 1) -
                                       RV.at(i, j, k - 1) * WW.at(i, j, k - 1)) *
                                      tmpz;
                }
            }
        }
        R3D_OMP_FOR
        for (int k = k1; k <= k2; ++k) {
            double tmpz = hz * s.dzzdz[k];
            for (int j = j1; j <= j2; ++j) {
                double tmpy = hy * s.dyydy[j];
                for (int i = i1; i <= i2; ++i) {
                    FW.at(i, j, k) -= (RW.at(i + 1, j, k) * UU.at(i + 1, j, k) -
                                       RW.at(i - 1, j, k) * UU.at(i - 1, j, k)) *
                                      hx * s.dxxdx[i];
                    FW.at(i, j, k) -= (RW.at(i, j + 1, k) * VV.at(i, j + 1, k) -
                                       RW.at(i, j - 1, k) * VV.at(i, j - 1, k)) *
                                      tmpy;
                    FW.at(i, j, k) -= (RW.at(i, j, k + 1) * WW.at(i, j, k + 1) -
                                       RW.at(i, j, k - 1) * WW.at(i, j, k - 1)) *
                                      tmpz;
                }
            }
        }

        // ===================================================================
        // Internal energy: advection then second-order diffusion of T.
        // ===================================================================
        R3D_OMP_FOR
        for (int k = k1; k <= k2; ++k) {
            double tmpz = hz * s.dzzdz[k];
            for (int j = j1; j <= j2; ++j) {
                double tmpy = hy * s.dyydy[j];
                for (int i = i1; i <= i2; ++i) {
                    FT.at(i, j, k) = -1.0 * UU.at(i, j, k) *
                                     (TT.at(i + 1, j, k) - TT.at(i - 1, j, k)) * hx * s.dxxdx[i];
                    FT.at(i, j, k) -=
                        VV.at(i, j, k) * (TT.at(i, j + 1, k) - TT.at(i, j - 1, k)) * tmpy;
                    FT.at(i, j, k) -=
                        WW.at(i, j, k) * (TT.at(i, j, k + 1) - TT.at(i, j, k - 1)) * tmpz;
                }
            }
        }

        // Thermal diffusion of T (non-LREM branch).
        {
            const double tmp = d.ocv / d.repr;
            R3D_OMP_FOR
            for (int k = k1; k <= k2; ++k) {
                const double tmpz1 = hz * s.d2zzdz2[k];
                const double tmpz2 = h2z * s.dzzdz[k] * s.dzzdz[k];
                for (int j = j1; j <= j2; ++j) {
                    const double tmpy1 = hy * s.d2yydy2[j];
                    const double tmpy2 = h2y * s.dyydy[j] * s.dyydy[j];
                    for (int i = i1; i <= i2; ++i) {
                        FT.at(i, j, k) +=
                            ((TT.at(i + 1, j, k) - TT.at(i - 1, j, k)) * hx * s.d2xxdx2[i] +
                             (TT.at(i + 1, j, k) - 2.0 * TT.at(i, j, k) + TT.at(i - 1, j, k)) *
                                 h2x * s.dxxdx[i] * s.dxxdx[i]) *
                            RO.at(i, j, k) * tmp * rkapa[k];
                        FT.at(i, j, k) +=
                            ((TT.at(i, j + 1, k) - TT.at(i, j - 1, k)) * tmpy1 +
                             (TT.at(i, j + 1, k) - 2.0 * TT.at(i, j, k) + TT.at(i, j - 1, k)) *
                                 tmpy2) *
                            RO.at(i, j, k) * tmp * rkapa[k];
                        FT.at(i, j, k) +=
                            ((TT.at(i, j, k + 1) - TT.at(i, j, k - 1)) * tmpz1 +
                             (TT.at(i, j, k + 1) - 2.0 * TT.at(i, j, k) + TT.at(i, j, k - 1)) *
                                 tmpz2) *
                            RO.at(i, j, k) * tmp * rkapa[k];
                        FT.at(i, j, k) += (TT.at(i, j, k + 1) - TT.at(i, j, k - 1)) * hz *
                                          s.dzzdz[k] * RO.at(i, j, k) * tmp * dkapa[k];
                    }
                }
            }
        }

        // ===================================================================
        // Viscous terms on the momenta.
        // ===================================================================
        R3D_OMP_FOR
        for (int k = k1; k <= k2; ++k) {
            const double tmpz1 = hz * s.d2zzdz2[k];
            const double tmpz2 = h2z * s.dzzdz[k] * s.dzzdz[k];
            for (int j = j1; j <= j2; ++j) {
                const double tmpy1 = hy * s.d2yydy2[j];
                const double tmpy2 = h2y * s.dyydy[j] * s.dyydy[j];
                for (int i = i1; i <= i2; ++i) {
                    FU.at(i, j, k) +=
                        ore *
                        (c43 * ((UU.at(i + 1, j, k) - UU.at(i - 1, j, k)) * hx * s.d2xxdx2[i] +
                                (UU.at(i + 1, j, k) - 2.0 * UU.at(i, j, k) + UU.at(i - 1, j, k)) *
                                    h2x * s.dxxdx[i] * s.dxxdx[i]) +
                         (UU.at(i, j + 1, k) - UU.at(i, j - 1, k)) * tmpy1 +
                         (UU.at(i, j + 1, k) - 2.0 * UU.at(i, j, k) + UU.at(i, j - 1, k)) * tmpy2 +
                         (UU.at(i, j, k + 1) - UU.at(i, j, k - 1)) * tmpz1 +
                         (UU.at(i, j, k + 1) - 2.0 * UU.at(i, j, k) + UU.at(i, j, k - 1)) * tmpz2);
                    FV.at(i, j, k) +=
                        ore *
                        (((VV.at(i + 1, j, k) - VV.at(i - 1, j, k)) * hx * s.d2xxdx2[i] +
                          (VV.at(i + 1, j, k) - 2.0 * VV.at(i, j, k) + VV.at(i - 1, j, k)) * h2x *
                              s.dxxdx[i] * s.dxxdx[i]) +
                         c43 * ((VV.at(i, j + 1, k) - VV.at(i, j - 1, k)) * tmpy1 +
                                (VV.at(i, j + 1, k) - 2.0 * VV.at(i, j, k) + VV.at(i, j - 1, k)) *
                                    tmpy2) +
                         (VV.at(i, j, k + 1) - VV.at(i, j, k - 1)) * tmpz1 +
                         (VV.at(i, j, k + 1) - 2.0 * VV.at(i, j, k) + VV.at(i, j, k - 1)) * tmpz2);
                    FW.at(i, j, k) +=
                        ore *
                        (((WW.at(i + 1, j, k) - WW.at(i - 1, j, k)) * hx * s.d2xxdx2[i] +
                          (WW.at(i + 1, j, k) - 2.0 * WW.at(i, j, k) + WW.at(i - 1, j, k)) * h2x *
                              s.dxxdx[i] * s.dxxdx[i]) +
                         (WW.at(i, j + 1, k) - WW.at(i, j - 1, k)) * tmpy1 +
                         (WW.at(i, j + 1, k) - 2.0 * WW.at(i, j, k) + WW.at(i, j - 1, k)) * tmpy2 +
                         c43 * ((WW.at(i, j, k + 1) - WW.at(i, j, k - 1)) * tmpz1 +
                                (WW.at(i, j, k + 1) - 2.0 * WW.at(i, j, k) + WW.at(i, j, k - 1)) *
                                    tmpz2));
                }
            }
        }

        // ===================================================================
        // Viscous heating, compressional heating and dissipation (FT).
        // dUU/dx etc.  Note the Fortran loops for WW1 (=dUU/dx), WW2 (=dVV/dx)
        // run over ALL j (J=1,NY) while WW3 (=dWW/dx) runs over interior j.
        // ===================================================================
        R3D_OMP_FOR
        for (int k = k1; k <= k2; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = i1; i <= i2; ++i)
                    WW1.at(i, j, k) = (UU.at(i + 1, j, k) - UU.at(i - 1, j, k)) * hx * s.dxxdx[i];
        R3D_OMP_FOR
        for (int k = k1; k <= k2; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = i1; i <= i2; ++i)
                    WW2.at(i, j, k) = (VV.at(i + 1, j, k) - VV.at(i - 1, j, k)) * hx * s.dxxdx[i];
        R3D_OMP_FOR
        for (int k = k1; k <= k2; ++k)
            for (int j = j1; j <= j2; ++j)
                for (int i = i1; i <= i2; ++i)
                    WW3.at(i, j, k) = (WW.at(i + 1, j, k) - WW.at(i - 1, j, k)) * hx * s.dxxdx[i];

        {
            const double tmp = d.ocv * ore;
            R3D_OMP_FOR
            for (int k = k1; k <= k2; ++k)
                for (int j = j1; j <= j2; ++j)
                    for (int i = i1; i <= i2; ++i)
                        FT.at(i, j, k) += (c43 * WW1.at(i, j, k) * WW1.at(i, j, k) +
                                           WW2.at(i, j, k) * WW2.at(i, j, k) +
                                           WW3.at(i, j, k) * WW3.at(i, j, k)) *
                                          RO.at(i, j, k) * tmp;
            R3D_OMP_FOR
            for (int k = k1; k <= k2; ++k)
                for (int j = j1; j <= j2; ++j)
                    for (int i = i1; i <= i2; ++i)
                        FT.at(i, j, k) -= d.ocv * WW1.at(i, j, k) * TT.at(i, j, k);
        }

        // Cross terms coupling the horizontal shear to FU/FV.
        {
            const double tmp = c13 * ore;
            R3D_OMP_FOR
            for (int k = k1; k <= k2; ++k)
                for (int j = j1; j <= j2; ++j)
                    for (int i = i1; i <= i2; ++i)
                        FU.at(i, j, k) +=
                            tmp * (WW2.at(i, j + 1, k) - WW2.at(i, j - 1, k)) * hy * s.dyydy[j];
            R3D_OMP_FOR
            for (int k = k1; k <= k2; ++k)
                for (int j = j1; j <= j2; ++j)
                    for (int i = i1; i <= i2; ++i)
                        FV.at(i, j, k) +=
                            tmp * (WW1.at(i, j + 1, k) - WW1.at(i, j - 1, k)) * hy * s.dyydy[j];
        }

        // dUU/dy, dVV/dy: dissipation terms from horizontal shear.
        R3D_OMP_FOR
        for (int k = k1; k <= k2; ++k)
            for (int j = j1; j <= j2; ++j)
                for (int i = i1; i <= i2; ++i)
                    WW1.at(i, j, k) = (UU.at(i, j + 1, k) - UU.at(i, j - 1, k)) * hy * s.dyydy[j];
        {
            const double tmp = d.ocv * ore;
            R3D_OMP_FOR
            for (int k = k1; k <= k2; ++k)
                for (int j = j1; j <= j2; ++j)
                    for (int i = i1; i <= i2; ++i)
                        FT.at(i, j, k) += (WW1.at(i, j, k) * WW1.at(i, j, k) +
                                           2.0 * WW1.at(i, j, k) * WW2.at(i, j, k)) *
                                          RO.at(i, j, k) * tmp;
        }
        R3D_OMP_FOR
        for (int k = k1; k <= k2; ++k)
            for (int j = j1; j <= j2; ++j)
                for (int i = i1; i <= i2; ++i)
                    WW2.at(i, j, k) = (VV.at(i, j + 1, k) - VV.at(i, j - 1, k)) * hy * s.dyydy[j];
        {
            const double tmp = c43 * d.ocv * ore;
            R3D_OMP_FOR
            for (int k = k1; k <= k2; ++k)
                for (int j = j1; j <= j2; ++j)
                    for (int i = i1; i <= i2; ++i)
                        FT.at(i, j, k) += WW2.at(i, j, k) * WW2.at(i, j, k) * RO.at(i, j, k) * tmp;
            R3D_OMP_FOR
            for (int k = k1; k <= k2; ++k)
                for (int j = j1; j <= j2; ++j)
                    for (int i = i1; i <= i2; ++i)
                        FT.at(i, j, k) -= d.ocv * WW2.at(i, j, k) * TT.at(i, j, k);
        }
        // Recompute WW1 = dUU/dx (used by the vertical dissipation below).
        R3D_OMP_FOR
        for (int k = k1; k <= k2; ++k)
            for (int j = j1; j <= j2; ++j)
                for (int i = i1; i <= i2; ++i)
                    WW1.at(i, j, k) = (UU.at(i + 1, j, k) - UU.at(i - 1, j, k)) * hx * s.dxxdx[i];

        // dWW/dz over the FULL i,j array.
        R3D_OMP_FOR
        for (int k = k1; k <= k2; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                    WW3.at(i, j, k) = (WW.at(i, j, k + 1) - WW.at(i, j, k - 1)) * hz * s.dzzdz[k];

        {
            const double tmp = c43 * d.ocv * ore;
            R3D_OMP_FOR
            for (int k = k1; k <= k2; ++k)
                for (int j = j1; j <= j2; ++j)
                    for (int i = i1; i <= i2; ++i)
                        FT.at(i, j, k) +=
                            (WW3.at(i, j, k) * WW3.at(i, j, k) - WW1.at(i, j, k) * WW3.at(i, j, k) -
                             WW1.at(i, j, k) * WW2.at(i, j, k) -
                             WW2.at(i, j, k) * WW3.at(i, j, k)) *
                            RO.at(i, j, k) * tmp;
            R3D_OMP_FOR
            for (int k = k1; k <= k2; ++k)
                for (int j = j1; j <= j2; ++j)
                    for (int i = i1; i <= i2; ++i)
                        FT.at(i, j, k) -= d.ocv * WW3.at(i, j, k) * TT.at(i, j, k);
        }

        {
            const double tmp = c13 * ore;
            R3D_OMP_FOR
            for (int k = k1; k <= k2; ++k)
                for (int j = j1; j <= j2; ++j)
                    for (int i = i1; i <= i2; ++i)
                        FU.at(i, j, k) +=
                            tmp * (WW3.at(i + 1, j, k) - WW3.at(i - 1, j, k)) * hx * s.dxxdx[i];
            R3D_OMP_FOR
            for (int k = k1; k <= k2; ++k)
                for (int j = j1; j <= j2; ++j)
                    for (int i = i1; i <= i2; ++i)
                        FV.at(i, j, k) +=
                            tmp * (WW3.at(i, j + 1, k) - WW3.at(i, j - 1, k)) * hy * s.dyydy[j];
        }

        // dUU/dz (I=1..NX all i), dVV/dz (J=1..NY all j).
        R3D_OMP_FOR
        for (int k = k1; k <= k2; ++k)
            for (int j = j1; j <= j2; ++j)
                for (int i = 0; i < nx; ++i)
                    WW1.at(i, j, k) = (UU.at(i, j, k + 1) - UU.at(i, j, k - 1)) * hz * s.dzzdz[k];
        R3D_OMP_FOR
        for (int k = k1; k <= k2; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = i1; i <= i2; ++i)
                    WW2.at(i, j, k) = (VV.at(i, j, k + 1) - VV.at(i, j, k - 1)) * hz * s.dzzdz[k];

        // dWW/dy (interior) then the final dissipation sum.
        R3D_OMP_FOR
        for (int k = k1; k <= k2; ++k)
            for (int j = j1; j <= j2; ++j)
                for (int i = i1; i <= i2; ++i)
                    WW3.at(i, j, k) = (WW.at(i, j + 1, k) - WW.at(i, j - 1, k)) * hy * s.dyydy[j];
        {
            const double tmp = d.ocv * ore;
            R3D_OMP_FOR
            for (int k = k1; k <= k2; ++k)
                for (int j = j1; j <= j2; ++j)
                    for (int i = i1; i <= i2; ++i)
                        FT.at(i, j, k) +=
                            (WW1.at(i, j, k) * WW1.at(i, j, k) + WW2.at(i, j, k) * WW2.at(i, j, k) +
                             WW3.at(i, j, k) * WW3.at(i, j, k) +
                             2.0 * WW2.at(i, j, k) * WW3.at(i, j, k)) *
                            RO.at(i, j, k) * tmp;
        }

        {
            const double tmp = c13 * ore;
            R3D_OMP_FOR
            for (int k = k1; k <= k2; ++k)
                for (int j = j1; j <= j2; ++j)
                    for (int i = i1; i <= i2; ++i)
                        FW.at(i, j, k) +=
                            tmp * ((WW1.at(i + 1, j, k) - WW1.at(i - 1, j, k)) * hx * s.dxxdx[i] +
                                   (WW2.at(i, j + 1, k) - WW2.at(i, j - 1, k)) * hy * s.dyydy[j]);
        }

        R3D_OMP_FOR
        for (int k = k1; k <= k2; ++k)
            for (int j = j1; j <= j2; ++j)
                for (int i = i1; i <= i2; ++i)
                    WW3.at(i, j, k) = (WW.at(i + 1, j, k) - WW.at(i - 1, j, k)) * hx * s.dxxdx[i];
        {
            const double tmp = d.ocv * ore;
            R3D_OMP_FOR
            for (int k = k1; k <= k2; ++k)
                for (int j = j1; j <= j2; ++j)
                    for (int i = i1; i <= i2; ++i)
                        FT.at(i, j, k) +=
                            2.0 * WW1.at(i, j, k) * WW3.at(i, j, k) * RO.at(i, j, k) * tmp;
        }

        // ===================================================================
        // Rotation (Coriolis) terms.
        // ===================================================================
        if (p.lrot) {
            R3D_OMP_FOR
            for (int k = k1; k <= k2; ++k)
                for (int j = j1; j <= j2; ++j)
                    for (int i = i1; i <= i2; ++i) FU.at(i, j, k) += d.omz * RV.at(i, j, k);
            R3D_OMP_FOR
            for (int k = k1; k <= k2; ++k)
                for (int j = j1; j <= j2; ++j)
                    for (int i = i1; i <= i2; ++i)
                        FV.at(i, j, k) += -d.omz * RU.at(i, j, k) + d.omx * RW.at(i, j, k);
            R3D_OMP_FOR
            for (int k = k1; k <= k2; ++k)
                for (int j = j1; j <= j2; ++j)
                    for (int i = i1; i <= i2; ++i) FW.at(i, j, k) += -d.omx * RV.at(i, j, k);
        }

        // ===================================================================
        // Magnetic fields (LMAG branch).  RU/RV/RW and TT are reused as scratch
        // for velocity/B derivatives; WW1..WW3 accumulate the B-field fluxes.
        // ===================================================================
        if (p.lmag) {
            unsupported("LMAG (magnetic branch not yet ported, coming with STEP/BCON)");
        }
    }

}  // namespace r3d
