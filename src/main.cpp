// main.cpp - the 3dmhd driver (port of 3dmhd.f).
//
// Mirrors the original flow: MPI init, processor-grid resolution (parsed,
// not hardcoded), grid Jacobians, thermal conductivity, STATIC initial
// conditions (or restart), one halo exchange, an initial timestep estimate,
// and the RK time loop with periodic dumps and a compact log.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>

#include "communicate.hpp"
#include "fields.hpp"
#include "grid.hpp"
#include "numcompat.hpp"
#include "params.hpp"
#include "static_ic.hpp"
#include "step.hpp"
#include "topology.hpp"

#ifdef R3D_HAVE_GPU
#include "gpu.hpp"
#endif

namespace r3d {

    // Print a message, flush it, then abort MPI.  Flushing matters: when
    // stderr is a buffered pipe (as under `mpirun ... 2>&1 | grep`), a
    // message left in the buffer vanishes if another rank's abort kills the
    // job first - which is exactly how the CPU-only "--backend gpu" guard
    // used to lose its diagnostic.
    [[noreturn]] void fatal(const std::string& msg) {
        fprintf(stderr, "%s\n", msg.c_str());
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Dump the full state in the golden gstate format (see read_gstate.py).
    // tag chooses the file set; iter suffixes the filename.
    void write_gstate(const Params& p, const Derived& d, const Topology& t, const SimState& s,
                      const char* tag, int iter) {
        char fname[256];
        snprintf(fname, sizeof(fname), "gstate.%s.%03d.%04d", tag, t.mype, iter);
        FILE* f = fopen(fname, "wb");
        if (!f) fatal("rast-3dmhd: cannot open dump '" + std::string(fname) + "'");
        auto w        = [&](const void* v, size_t n) { fwrite(v, 1, n, f); };
        int32_t magic = 1129928788, nf = 27, n1 = 14;
        w(&magic, 4);
        w(&nf, 4);
        w(&n1, 4);
        int32_t hdr[13] = {d.nx,
                           d.ny,
                           d.nz,
                           t.mype,
                           t.mypey,
                           t.mypez,
                           s.nit,
                           (int32_t) p.lmag,
                           (int32_t) p.ixcon,
                           (int32_t) p.iycon,
                           (int32_t) p.izcon,
                           (int32_t) p.itcon,
                           (int32_t) p.ibcon};
        w(hdr, sizeof(hdr));
        double sc[27] = {s.dt,  s.timc,  s.timt, s.timi,   s.umach, p.re, d.repr,  d.cv,  d.ocv,
                         d.ore, p.theta, p.grav, d.rkapst, d.orm,   p.rm, d.obeta, d.omx, d.omz,
                         p.sf,  p.ampt,  0.0,    d.hx,     d.h2x,   d.hy, d.h2y,   d.hz,  d.h2z};
        w(sc, sizeof(sc));
        const Field* f3d[27] = {&s.ru,  &s.rv,  &s.rw,  &s.ro,  &s.tt,  &s.uu,  &s.vv,
                                &s.ww,  &s.fu,  &s.fv,  &s.fw,  &s.fr,  &s.ft,  &s.zru,
                                &s.zrv, &s.zrw, &s.zro, &s.ztt, &s.ww1, &s.ww2, &s.ww3,
                                &s.bx,  &s.by,  &s.bz,  &s.zbx, &s.zby, &s.zbz};
        for (int i = 0; i < 27; ++i) {
            int64_t cnt = (int64_t) d.nx * d.ny * d.nz;
            w(&cnt, 8);
            w(f3d[i]->data(), cnt * 8);
        }
        const std::vector<double>* f1d[14] = {&s.exx,     &s.dxxdx,   &s.d2xxdx2, &s.ddx,  &s.wyy,
                                              &s.dyydy,   &s.d2yydy2, &s.ddy,     &s.zee,  &s.dzzdz,
                                              &s.d2zzdz2, &s.ddz,     &s.rkapa,   &s.dkapa};
        for (int i = 0; i < 14; ++i) {
            int64_t cnt = f1d[i]->size();
            w(&cnt, 8);
            w(f1d[i]->data(), cnt * 8);
        }
        int64_t end = -1;
        w(&end, 8);
        fclose(f);
    }

    namespace {
        // Initial explicit timestep estimate (main's ISW=1 branch); only used for
        // the startup log since STEP recomputes dt on every call.
        double initial_dt(const Params& p, const Derived& d, const Topology& t, const SimState& s) {
            const int nx = d.nx, ny = d.ny, nz = d.nz;
            const int k1 = p.ilap / 2, k2 = nz - 1 - p.ilap / 2;
            double wmin[3] = {1e300, 1e300, 1e300};
            for (int k = k1; k <= k2; ++k)
                for (int j = 1; j <= ny - p.iy; ++j)
                    for (int i = 1; i <= nx - p.ix; ++i) {
                        double dd = std::min(std::min(s.ddx[i], s.ddy[j]), s.ddz[k]);
                        wmin[0]   = std::min(wmin[0], dd / std::sqrt(s.uu.at(i, j, k)));
                        wmin[1]   = std::min(
                            wmin[1], 0.5 * dd * dd * d.repr * d.cv * s.ro.at(i, j, k) / s.rkapa[k]);
                        wmin[2] = std::min(wmin[2], 0.375 * dd * dd * p.re * s.ro.at(i, j, k));
                    }
            double wout[3];
            MPI_Allreduce(wmin, wout, 3, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            return p.sf * std::min(std::min(wout[0], wout[1]), wout[2]);
        }
    }  // namespace

}  // namespace r3d

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int npe, mype;
    MPI_Comm_size(MPI_COMM_WORLD, &npe);
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);

    r3d::Params p;
    if (r3d::parse_args(argc, argv, p) != 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    r3d::Topology t = r3d::resolve_topology(npe, mype, p);
    r3d::Derived d  = r3d::derive(p, npe, t.npey, t.npez);
    r3d::SimState s(d);

    // Grid Jacobians + radiatiove conductivity.
    r3d::build_grid_metrics(p, t, s);
    if (p.pzp == 0.0) {
        std::fill(s.rkapa.begin(), s.rkapa.end(), 1.0);
        std::fill(s.dkapa.begin(), s.dkapa.end(), 0.0);
    } else {
        for (int k = 0; k < d.nz; ++k) {
            double xe  = s.zee[k];
            s.rkapa[k] = 1.0 + (d.rkapst - 1.0) / 2.0 * (1.0 + std::tanh((xe - p.pzp) / p.sigma));
            s.dkapa[k] = (d.rkapst - 1.0) / 2.0 / p.sigma / std::cosh((xe - p.pzp) / p.sigma) /
                         std::cosh((xe - p.pzp) / p.sigma);
            // NOTE: the original DKAPA = .../COSH(...)**2, a single tanh model.
        }
    }

    if (p.nstart == 0) {
        r3d::static_ic(MPI_COMM_WORLD, p, d, t, s);
        s.timi = 0.0;
    } else {
        r3d::fatal("rast-3dmhd: restart (NSTART!=0) is not yet ported");
    }
    r3d::communicate(MPI_COMM_WORLD, p, t, s);

    // Velocities and the startup timestep estimate (for the log).
    const int k1 = p.ilap / 2, k2 = d.nz - 1 - p.ilap / 2;
    for (int k = k1; k <= k2; ++k)
        for (int j = 1; j <= d.ny - p.iy; ++j)
            for (int i = 1; i <= d.nx - p.ix; ++i)
                s.uu.at(i, j, k) =
                    (s.ru.at(i, j, k) * s.ru.at(i, j, k) + s.rv.at(i, j, k) * s.rv.at(i, j, k) +
                     s.rw.at(i, j, k) * s.rw.at(i, j, k)) *
                        (1.0 / s.ro.at(i, j, k)) * (1.0 / s.ro.at(i, j, k)) +
                    p.gamma * s.tt.at(i, j, k) +
                    2.0 * std::sqrt((s.ru.at(i, j, k) * s.ru.at(i, j, k) +
                                     s.rv.at(i, j, k) * s.rv.at(i, j, k) +
                                     s.rw.at(i, j, k) * s.rw.at(i, j, k)) *
                                    (1.0 / s.ro.at(i, j, k)) * (1.0 / s.ro.at(i, j, k)) * p.gamma *
                                    s.tt.at(i, j, k));
    double dt0 = r3d::initial_dt(p, d, t, s);

    if (mype == 0) {
        fprintf(stderr,
                "rast-3dmhd: NP=%d NPEY=%d NPEZ=%d  local %dx%dx%d  "
                "NIT0=%d DT=%g NTOTAL=%d\n",
                npe, t.npey, t.npez, d.nx, d.ny, d.nz, p.ntotal, dt0, p.ntotal);
    }

    // ------------------------------------------------------------------
    // Compile-time dump mode (set via -DR3D_DUMP_MODE=N in the makefile):
    //   0 = none     - no gstate dump files at all (progress is still logged)
    //   1 = final    - the initial gstate.start plus only the final step
    //   2 = periodic - NSTEP0-cadence dumps plus the final step (default)
    // Each dump is ~NX*NY*NZ*27*8 B per rank (e.g. ~118 GB per checkpoint at
    // the default 504x504x2048 grid), so the more restrictive modes keep long
    // runs from filling the filesystem.
    // ------------------------------------------------------------------
#ifndef R3D_DUMP_MODE
#define R3D_DUMP_MODE 2
#endif
#if R3D_DUMP_MODE < 0 || R3D_DUMP_MODE > 2
#error "R3D_DUMP_MODE must be 0 (none), 1 (final) or 2 (periodic)"
#endif

    // The initial-condition dump (gstate.start.NIT0) is written in every mode
    // except "none".
#if R3D_DUMP_MODE != 0
    r3d::write_gstate(p, d, t, s, "start", 0);
#endif

    // Wall-clock timing for the progress log (CLOCK_MONOTONIC: immune to NTP).
    struct timespec wstart;
    clock_gettime(CLOCK_MONOTONIC, &wstart);
    auto wall_since = [&]() {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        return (now.tv_sec - wstart.tv_sec) + 1e-9 * (now.tv_nsec - wstart.tv_nsec);
    };

    // Whether this step writes a gstate dump.
    auto should_dump = [&](int nit, int nk) {
#if R3D_DUMP_MODE == 0
        (void) nit;
        return false;
#elif R3D_DUMP_MODE == 1
        (void) nit;
        return nk == p.ntotal;
#else
        return (nit % p.nstep0 == 0) || nk == p.ntotal;
#endif
    };
    // Whether this step prints the NIT/timing progress line.  Every mode logs
    // at the NSTEP0 cadence (and at the end) so even a dump-less run stays
    // observable; in the dumping modes the progress line accompanies each dump.
    auto should_log = [&](int nit, int nk) {
#if R3D_DUMP_MODE == 0
        return (nit % p.nstep0 == 0) || nk == p.ntotal;
#else
        return should_dump(nit, nk);
#endif
    };

#ifdef R3D_HAVE_GPU
    if (p.backend == r3d::Params::Backend::kGpu) {
        // GPU-resident time loop: the full state stays on the device; only the
        // few per-step reduction scalars and the periodic dumps cross the bus.
        cudaSetDevice(p.gpu_id);  // pick this rank's GPU (honours CUDA_VISIBLE_DEVICES)
        r3d::GpuSim gsim;
        r3d::gpu_alloc(gsim, d);
        r3d::gpu_upload(gsim, s);
        r3d::SimState sc(d);  // scalar mirror: dt/nit/timc live here on the GPU path
        sc.tb    = s.tb;
        sc.timi  = s.timi;
        sc.timc  = s.timc;
        sc.timt  = s.timt;
        sc.nit   = s.nit;
        sc.dt    = s.dt;
        sc.umach = s.umach;
        for (int nk = 1; nk <= p.ntotal; ++nk) {
            r3d::step_gpu(MPI_COMM_WORLD, p, d, t, gsim, sc);
            if (sc.dt < 1.0e-8) {
                if (mype == 0)
                    fprintf(stderr, "rast-3dmhd: timestep underflow at NIT=%d\n", sc.nit);
                break;
            }
            if (should_log(sc.nit, nk)) {
                if (should_dump(sc.nit, nk)) {
                    r3d::gpu_download(s, gsim);
                    s.nit  = sc.nit;
                    s.dt   = sc.dt;
                    s.timc = sc.timc;
                    s.timt = sc.timt;
                    r3d::write_gstate(p, d, t, s, "step", s.nit);
                }
                if (mype == 0) {
                    double wall = wall_since();
                    fprintf(stderr, "NIT %d  DT %g  TIMC %g  UMACH %g  WALL %.1fs (%.2fs/step)\n",
                            sc.nit, sc.dt, sc.timc, sc.umach, wall, wall / sc.nit);
                }
            }
        }
        r3d::gpu_download(s, gsim);
        gpu_free(gsim);
    } else
#else
    if (p.backend == r3d::Params::Backend::kGpu) {
        // Fail loudly instead of silently running the CPU path on a CPU-only
        // build (the GPU scripts guard against this, but be safe anyway).
        if (mype == 0)
            r3d::fatal("rast-3dmhd: --backend gpu requested but this binary was built without CUDA "
                  "(rebuild with: make BACKEND=gpu)");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
#endif
    {
        // CPU-resident time loop.
        for (int nk = 1; nk <= p.ntotal; ++nk) {
            r3d::step(MPI_COMM_WORLD, p, d, t, s);
            if (s.dt < 1.0e-8) {
                if (mype == 0) fprintf(stderr, "rast-3dmhd: timestep underflow at NIT=%d\n", s.nit);
                break;
            }
            if (should_log(s.nit, nk)) {
                if (should_dump(s.nit, nk)) {
                    r3d::write_gstate(p, d, t, s, "step", s.nit);
                }
                if (mype == 0) {
                    double wall = wall_since();
                    fprintf(stderr, "NIT %d  DT %g  TIMC %g  UMACH %g  WALL %.1fs (%.2fs/step)\n",
                            s.nit, s.dt, s.timc, s.umach, wall, wall / s.nit);
                }
            }
        }
    }

    MPI_Finalize();
    return 0;
}
