// static_ic.cpp - initial conditions (port of 3dmhdsub.f STATIC).
#include "static_ic.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "numcompat.hpp"
#include "ode.hpp"
#include "ran2.hpp"

namespace r3d {

    namespace {
        [[noreturn]] void unsupported(const char* what) {
            fprintf(stderr, "rast-3dmhd: STATIC: unsupported feature enabled: %s\n", what);
            exit(EXIT_FAILURE);
        }
    }  // namespace

    void static_ic(MPI_Comm comm, const Params& p, const Derived& d, const Topology& t,
                   SimState& s) {
        if (p.lshr) unsupported("LSHR");
        if (p.lrem) unsupported("LREM");
        if (p.lmag && p.ntube != 0) unsupported("TUBE (LMAG tube models)");
        if (p.ampb != 0.0 && !p.lmag) unsupported("AMPB magnetic layer requires LMAG");

        const int nx = d.nx, ny = d.ny, nz = d.nz;
        const int ilaph = p.ilap / 2;
        const int iyx = p.iy / 2, ixx = p.ix / 2;

        // ------------------------------------------------------------------
        // Zero the velocity fields.
        // ------------------------------------------------------------------
        s.ru.fill(0.0);
        s.rv.fill(0.0);
        s.rw.fill(0.0);

        // ------------------------------------------------------------------
        // Vertical stratification (non-LREM): T/R profile via Bulirsch-Stoer.
        // ------------------------------------------------------------------
        std::vector<double> T(p.npz), R(p.npz);
        static_profile(p, p.npz, T.data(), R.data());
        const double tb = T[p.npz - 1];
        const double rb = R[p.npz - 1];
        s.tb            = tb;

        // Distribute the (i,j,all-k) columns to this rank's subdomain, exactly as
        // the Fortran slices T(1:NRZ+ILAP/2) etc.
        if (t.mypez == 0) {
            // k in [0, ilaph) = T(1) (2D spread); k in [ilaph, nz) = T[0..nrz+ilaph-1]
            for (int k = 0; k < ilaph; ++k)
                for (int j = 1; j <= ny - p.iy; ++j)
                    for (int i = 1; i <= nx - p.ix; ++i) {
                        s.tt.at(i, j, k) = T[0];
                        s.ro.at(i, j, k) = R[0];
                    }
            for (int k = ilaph; k < nz; ++k)
                for (int j = 1; j <= ny - p.iy; ++j)
                    for (int i = 1; i <= nx - p.ix; ++i) {
                        s.tt.at(i, j, k) = T[k - ilaph];
                        s.ro.at(i, j, k) = R[k - ilaph];
                    }
        } else if (t.mypez == t.npez - 1) {
            // k in [0, nrz+ilaph) = T[NPZ-NRZ-ilaph .. NPZ-1]; rest = TB/RB
            for (int k = 0; k < d.nrz + ilaph; ++k)
                for (int j = 1; j <= ny - p.iy; ++j)
                    for (int i = 1; i <= nx - p.ix; ++i) {
                        s.tt.at(i, j, k) = T[p.npz - d.nrz - ilaph + k];
                        s.ro.at(i, j, k) = R[p.npz - d.nrz - ilaph + k];
                    }
            for (int k = d.nrz + ilaph; k < nz; ++k)
                for (int j = 1; j <= ny - p.iy; ++j)
                    for (int i = 1; i <= nx - p.ix; ++i) {
                        s.tt.at(i, j, k) = tb;
                        s.ro.at(i, j, k) = rb;
                    }
        } else {
            const int i1 = t.mypez * d.nrz - ilaph;  // 0-based T offset
            for (int k = 0; k < nz; ++k)
                for (int j = 1; j <= ny - p.iy; ++j)
                    for (int i = 1; i <= nx - p.ix; ++i) {
                        s.tt.at(i, j, k) = T[i1 + k];
                        s.ro.at(i, j, k) = R[i1 + k];
                    }
        }

        // ------------------------------------------------------------------
        // Optional magnetic layer from a horizontal layer of B (LMAG).
        // ------------------------------------------------------------------
        if (p.lmag) {
            double cln = f4(-4.0f * logf(2.0f)) / (p.bfh * p.bfh);
            for (int k = ilaph; k <= nz - 1 - ilaph; ++k) {
                double bfac = p.ampb * std::exp(cln * (s.zee[k] - p.bzp) * (s.zee[k] - p.bzp));
                for (int j = 0; j < ny; ++j)
                    for (int i = 0; i < nx; ++i) s.bx.at(i, j, k) = bfac;
            }
            s.by.fill(0.0);
            s.bz.fill(0.0);
            for (int k = ilaph; k <= nz - 1 - ilaph; ++k)
                for (int j = 1; j <= ny - p.iy; ++j)
                    for (int i = 1; i <= nx - p.ix; ++i)
                        s.ro.at(i, j, k) = s.ro.at(i, j, k) - s.bx.at(i, j, k) * s.bx.at(i, j, k) /
                                                                  s.tt.at(i, j, k) * d.obeta;
        }

        // ------------------------------------------------------------------
        // Temperature perturbations (ISW=1 random branch).
        // Rank 0 draws RND with RAN2 and fans it out; every rank applies it.
        // ------------------------------------------------------------------
        if (p.ampt != 0.0) {
            // Interior cells the perturbation covers (1-based IX/2+1..NX-IX/2 etc).
            const int ip1 = ixx, ip2 = nx - ixx - 1;  // 0-based i range
            const int jp1 = iyx, jp2 = ny - iyx - 1;  // 0-based j range
            Ran2State rnd_state;
            rnd_state.idum = -62659;
            bool seeded    = false;
            std::vector<double> rnd((size_t) nx * ny, 0.0);
            for (int nnz = 1; nnz <= t.npez; ++nnz) {
                for (int k = ilaph; k <= nz - 1 - ilaph; ++k) {
                    const int itag = k;
                    for (int nny = 1; nny <= t.npey; ++nny) {
                        if (t.mype == 0) {
                            for (int jj = jp1; jj <= jp2; ++jj)
                                for (int ii = ip1; ii <= ip2; ++ii)
                                    rnd[(size_t) jj * nx + ii] =
                                        1.0 + p.ampt * (ran2(rnd_state) - 0.5);
                            const int target = nny + t.npey * (nnz - 1) - 1;
                            if (target != 0) {
                                // Send the full (nx*ny) plane to the target rank.
                                // Original: RND(I,J) for I=2..NX-IX/2 etc is set; the full
                                // NX*NY plane is sent.
                                MPI_Send(rnd.data(), nx * ny, MPI_DOUBLE, target, itag, comm);
                            } else {
                                // Keep the plane as WW1(:,:,k).
                                for (int jj = 0; jj < ny; ++jj)
                                    for (int ii = 0; ii < nx; ++ii)
                                        s.ww1.at(ii, jj, k) = rnd[(size_t) jj * nx + ii];
                            }
                        } else if (t.mype == nny + t.npey * (nnz - 1) - 1) {
                            std::vector<double> plane((size_t) nx * ny);
                            MPI_Recv(plane.data(), nx * ny, MPI_DOUBLE, 0, itag, comm,
                                     MPI_STATUS_IGNORE);
                            for (int jj = 0; jj < ny; ++jj)
                                for (int ii = 0; ii < nx; ++ii)
                                    s.ww1.at(ii, jj, k) = plane[(size_t) jj * nx + ii];
                            seeded = true;
                        }
                    }
                }
            }
            (void) seeded;
            // Apply the perturbation: TT = TT * WW1 over the covered cells.
            for (int k = ilaph; k <= nz - 1 - ilaph; ++k)
                for (int j = jp1; j <= jp2; ++j)
                    for (int i = ip1; i <= ip2; ++i) s.tt.at(i, j, k) *= s.ww1.at(i, j, k);
        }
    }

}  // namespace r3d
