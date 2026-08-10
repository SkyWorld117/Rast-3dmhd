// step.cpp - one RK3 time step (port of 3dmhdsub.f STEP).
#include "step.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "bcon.hpp"
#include "communicate.hpp"
#include "fluxes.hpp"

// Parallel sweeps with OpenMP (CPU backend only).  Every sweep writes only its
// own cell (independent stencil), and the per-step reductions below are exact
// min/max, so threading preserves the golden bit-exact result.  Compiled out
// when _OPENMP is undefined (the GPU backend builds these files as host code
// without -fopenmp).
#ifdef _OPENMP
#define R3D_OMP_FOR _Pragma("omp parallel for schedule(static)")
#else
#define R3D_OMP_FOR
#endif

namespace r3d {

    namespace {
        // Interior index ranges (0-based; see fluxes.cpp for the mapping).
        struct Int3 {
            int i1, i2, j1, j2, k1, k2;
        };
        Int3 interior(const Params& p, const Derived& d) {
            return {1, d.nx - p.ix, 1, d.ny - p.iy, p.ilap / 2, d.nz - 1 - p.ilap / 2};
        }

        // Copy current physical state into the Z* save buffers (interior only).
        void save_state(const Params& p, const Derived& d, SimState& s) {
            Int3 r = interior(p, d);
            R3D_OMP_FOR
            for (int k = r.k1; k <= r.k2; ++k)
                for (int j = r.j1; j <= r.j2; ++j)
                    for (int i = r.i1; i <= r.i2; ++i) {
                        s.zru.at(i, j, k) = s.ru.at(i, j, k);
                        s.zrv.at(i, j, k) = s.rv.at(i, j, k);
                        s.zrw.at(i, j, k) = s.rw.at(i, j, k);
                        s.zro.at(i, j, k) = s.ro.at(i, j, k);
                        s.ztt.at(i, j, k) = s.tt.at(i, j, k);
                    }
        }

        // One RK substep accumulator update: X = Z + COEF*F for the 5 primaries.
        void update_prim(const Params& p, const Derived& d, double coef, SimState& s) {
            Int3 r = interior(p, d);
            R3D_OMP_FOR
            for (int k = r.k1; k <= r.k2; ++k)
                for (int j = r.j1; j <= r.j2; ++j)
                    for (int i = r.i1; i <= r.i2; ++i)
                        s.ru.at(i, j, k) = s.zru.at(i, j, k) + coef * s.fu.at(i, j, k);
            R3D_OMP_FOR
            for (int k = r.k1; k <= r.k2; ++k)
                for (int j = r.j1; j <= r.j2; ++j)
                    for (int i = r.i1; i <= r.i2; ++i)
                        s.rv.at(i, j, k) = s.zrv.at(i, j, k) + coef * s.fv.at(i, j, k);
            R3D_OMP_FOR
            for (int k = r.k1; k <= r.k2; ++k)
                for (int j = r.j1; j <= r.j2; ++j)
                    for (int i = r.i1; i <= r.i2; ++i)
                        s.rw.at(i, j, k) = s.zrw.at(i, j, k) + coef * s.fw.at(i, j, k);
            R3D_OMP_FOR
            for (int k = r.k1; k <= r.k2; ++k)
                for (int j = r.j1; j <= r.j2; ++j)
                    for (int i = r.i1; i <= r.i2; ++i)
                        s.ro.at(i, j, k) = s.zro.at(i, j, k) + coef * s.fr.at(i, j, k);
            R3D_OMP_FOR
            for (int k = r.k1; k <= r.k2; ++k)
                for (int j = r.j1; j <= r.j2; ++j)
                    for (int i = r.i1; i <= r.i2; ++i)
                        s.tt.at(i, j, k) = s.ztt.at(i, j, k) + coef * s.ft.at(i, j, k);
        }

        // X = X + ZETA*F  (used between substeps: Z = U + zeta*dt*F)
        void mix_state(const Params& p, const Derived& d, double coef, SimState& s) {
            Int3 r = interior(p, d);
            R3D_OMP_FOR
            for (int k = r.k1; k <= r.k2; ++k)
                for (int j = r.j1; j <= r.j2; ++j)
                    for (int i = r.i1; i <= r.i2; ++i)
                        s.zru.at(i, j, k) = s.ru.at(i, j, k) + coef * s.fu.at(i, j, k);
            R3D_OMP_FOR
            for (int k = r.k1; k <= r.k2; ++k)
                for (int j = r.j1; j <= r.j2; ++j)
                    for (int i = r.i1; i <= r.i2; ++i)
                        s.zrv.at(i, j, k) = s.rv.at(i, j, k) + coef * s.fv.at(i, j, k);
            R3D_OMP_FOR
            for (int k = r.k1; k <= r.k2; ++k)
                for (int j = r.j1; j <= r.j2; ++j)
                    for (int i = r.i1; i <= r.i2; ++i)
                        s.zrw.at(i, j, k) = s.rw.at(i, j, k) + coef * s.fw.at(i, j, k);
            R3D_OMP_FOR
            for (int k = r.k1; k <= r.k2; ++k)
                for (int j = r.j1; j <= r.j2; ++j)
                    for (int i = r.i1; i <= r.i2; ++i)
                        s.zro.at(i, j, k) = s.ro.at(i, j, k) + coef * s.fr.at(i, j, k);
            R3D_OMP_FOR
            for (int k = r.k1; k <= r.k2; ++k)
                for (int j = r.j1; j <= r.j2; ++j)
                    for (int i = r.i1; i <= r.i2; ++i)
                        s.ztt.at(i, j, k) = s.tt.at(i, j, k) + coef * s.ft.at(i, j, k);
        }

    }  // namespace

    void step(int nstages, MPI_Comm comm, const Params& p, const Derived& d, const Topology& t,
              SimState& s) {
        Int3 r = interior(p, d);
        (void) r;
        const double sff    = p.sf;
        const double ogamma = 1.0 / p.gamma;

        // ---- velocity magnitudes, Mach number --------------------------------
        s.umach     = 0.0;
        double rmin = 1.0e9;
        double vmax = 0.0;
        R3D_OMP_FOR
        for (int k = r.k1; k <= r.k2; ++k)
            for (int j = r.j1; j <= r.j2; ++j)
                for (int i = r.i1; i <= r.i2; ++i)
                    s.uu.at(i, j, k) =
                        (s.ru.at(i, j, k) * s.ru.at(i, j, k) + s.rv.at(i, j, k) * s.rv.at(i, j, k) +
                         s.rw.at(i, j, k) * s.rw.at(i, j, k)) *
                        (1.0 / s.ro.at(i, j, k)) * (1.0 / s.ro.at(i, j, k));
        double umach_tmp = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(max : umach_tmp)
#endif
        for (int k = r.k1; k <= r.k2; ++k)
            for (int j = r.j1; j <= r.j2; ++j)
                for (int i = r.i1; i <= r.i2; ++i)
                    umach_tmp = std::max(umach_tmp, ogamma * s.uu.at(i, j, k) / s.tt.at(i, j, k));
        s.umach = std::sqrt(umach_tmp);

        // Fast-mode speed (with optional magnetic contribution).
        if (p.lmag) {
            R3D_OMP_FOR
            for (int k = r.k1; k <= r.k2; ++k)
                for (int j = r.j1; j <= r.j2; ++j)
                    for (int i = r.i1; i <= r.i2; ++i) {
                        double b2 = (s.bx.at(i, j, k) * s.bx.at(i, j, k) +
                                     s.by.at(i, j, k) * s.by.at(i, j, k) +
                                     s.bz.at(i, j, k) * s.bz.at(i, j, k)) *
                                    d.obeta / s.ro.at(i, j, k);
                        s.uu.at(i, j, k) =
                            s.uu.at(i, j, k) + p.gamma * s.tt.at(i, j, k) + b2 +
                            2.0 * std::sqrt(s.uu.at(i, j, k) * (p.gamma * s.tt.at(i, j, k) + b2));
                    }
        } else {
            R3D_OMP_FOR
            for (int k = r.k1; k <= r.k2; ++k)
                for (int j = r.j1; j <= r.j2; ++j)
                    for (int i = r.i1; i <= r.i2; ++i)
                        s.uu.at(i, j, k) =
                            s.uu.at(i, j, k) + p.gamma * s.tt.at(i, j, k) +
                            2.0 * std::sqrt(s.uu.at(i, j, k) * p.gamma * s.tt.at(i, j, k));
        }
        double vmax_tmp = 0.0, rmin_tmp = 1.0e9;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(max : vmax_tmp) reduction(min : rmin_tmp)
#endif
        for (int k = r.k1; k <= r.k2; ++k)
            for (int j = r.j1; j <= r.j2; ++j)
                for (int i = r.i1; i <= r.i2; ++i) {
                    vmax_tmp = std::max(vmax_tmp, s.uu.at(i, j, k));
                    rmin_tmp = std::min(rmin_tmp, s.ro.at(i, j, k));
                }
        vmax = std::sqrt(vmax_tmp);
        (void) rmin_tmp;

        double umach_all;
        MPI_Allreduce(&s.umach, &umach_all, 1, MPI_DOUBLE, MPI_MAX, comm);
        s.umach = umach_all;

        // ---- pointwise minimum timestep (ISW=1 branch) ------------------------
        // DD is the smallest grid spacing (pointwise), WW1..3 carry the per-cell
        // advective / diffusive / viscous limits.
        R3D_OMP_FOR
        for (int k = r.k1; k <= r.k2; ++k)
            for (int j = r.j1; j <= r.j2; ++j)
                for (int i = r.i1; i <= r.i2; ++i) {
                    double dd         = std::min(std::min(s.ddx[i], s.ddy[j]), s.ddz[k]);
                    s.ww1.at(i, j, k) = dd / std::sqrt(s.uu.at(i, j, k));
                    s.ww2.at(i, j, k) =
                        0.5 * dd * dd * d.repr * d.cv * s.ro.at(i, j, k) / s.rkapa[k];
                    s.ww3.at(i, j, k) = 0.375 * dd * dd * p.re * s.ro.at(i, j, k);
                    if (p.lmag) s.vv.at(i, j, k) = 0.5 * dd * dd * p.rm;
                }
        double wmin[4], wout[4];
        int mincnt = 3;
        double w0 = 1e300, w1 = 1e300, w2 = 1e300;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(min : w0, w1, w2)
#endif
        for (int k = r.k1; k <= r.k2; ++k)
            for (int j = r.j1; j <= r.j2; ++j)
                for (int i = r.i1; i <= r.i2; ++i) {
                    w0 = std::min(w0, s.ww1.at(i, j, k));
                    w1 = std::min(w1, s.ww2.at(i, j, k));
                    w2 = std::min(w2, s.ww3.at(i, j, k));
                }
        wmin[0] = w0;
        wmin[1] = w1;
        wmin[2] = w2;
        if (p.lmag) {
            double w3 = 1e300;
            mincnt    = 4;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(min : w3)
#endif
            for (int k = r.k1; k <= r.k2; ++k)
                for (int j = r.j1; j <= r.j2; ++j)
                    for (int i = r.i1; i <= r.i2; ++i) w3 = std::min(w3, s.vv.at(i, j, k));
            wmin[3] = w3;
        }
        MPI_Allreduce(wmin, wout, mincnt, MPI_DOUBLE, MPI_MIN, comm);
        if (p.lmag) {
            s.dt = sff * std::min(std::min(std::min(wout[0], wout[1]), wout[2]), wout[3]);
        } else {
            s.dt = sff * std::min(std::min(wout[0], wout[1]), wout[2]);
        }

        // ---- low-storage Wray RK3 ------------------------------------------
        save_state(p, d, s);
        for (int st = 0; st < nstages; ++st) {
            if (st == 0) {
                fluxes(p, d, t, s.rkapa.data(), s.dkapa.data(), s);
                update_prim(p, d, d.gam1 * s.dt, s);
                bcon(p, d, t, s);
                communicate(comm, p, t, s);
            } else if (st == 1) {
                mix_state(p, d, d.zeta1 * s.dt, s);
                fluxes(p, d, t, s.rkapa.data(), s.dkapa.data(), s);
                update_prim(p, d, d.gam2 * s.dt, s);
                bcon(p, d, t, s);
                communicate(comm, p, t, s);
            } else {
                mix_state(p, d, d.zeta2 * s.dt, s);
                fluxes(p, d, t, s.rkapa.data(), s.dkapa.data(), s);
                update_prim(p, d, d.gam3 * s.dt, s);
                bcon(p, d, t, s);
                communicate(comm, p, t, s);
            }
        }

        s.nit += 1;
        s.timc += s.dt;
        s.timt = s.timi + s.timc;
    }

    void step(MPI_Comm comm, const Params& p, const Derived& d, const Topology& t, SimState& s) {
        step(3, comm, p, d, t, s);
    }

}  // namespace r3d
