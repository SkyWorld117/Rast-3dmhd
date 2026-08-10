// gpu.cpp - device-resident simulation state and the GPU step driver.
//
// step_gpu() mirrors step()/fluxes()/bcon()/communicate() (step.cpp,
// fluxes.cpp, bcon.cpp, communicate.cpp) but every compute sweep runs on the
// device via the kernels in gpu_kernels.cu; only the few reduction scalars
// cross the PCIe bus each step.  The halo exchange uses CUDA-aware MPI on
// device pointers; strided y/x slabs are packed through small device
// staging buffers.
#include "gpu.hpp"

#include <cuda_runtime.h>
#include <mpi.h>

#ifdef R3D_COMM_NCCL
#include <nccl.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gpu_kernels.hpp"
#include "numcompat.hpp"

namespace r3d {

    // ---------------------------------------------------------------------------
    // Kernel prototypes (defined in gpu_kernels.cu).
    // ---------------------------------------------------------------------------
    __global__ void k_fr(const K, const double*, const double*, const double*, double*, double*,
                         const double*, const double*, const double*);
    __global__ void k_fr_w1b(const K, const double*, double*, const double*, int);
    __global__ void k_fr_subw1(const K, double*, const double*);
    __global__ void k_force_prep(const K, const double*, const double*, double*, double*);
    __global__ void k_force(const K, double, const double*, double*, double*, double*,
                            const double*, const double*, const double*, const double*);
    __global__ void k_vel(const K, const double*, const double*, const double*, const double*,
                          double*, double*, double*);
    __global__ void k_adv(const K, int, const double*, const double*, const double*, const double*,
                          const double*, const double*, double*, double*, double*, const double*,
                          const double*, const double*);
    __global__ void k_ft_advect(const K, const double*, const double*, const double*, const double*,
                                double*, const double*, const double*, const double*);
    __global__ void k_ft_diffuse(const K, const double*, const double*, double*, const double*,
                                 const double*, const double*, const double*, const double*,
                                 const double*, const double*, const double*);
    __global__ void k_viscous_fu(const K, const double*, const double*, const double*, double*,
                                 double*, double*, const double*, const double*, const double*,
                                 const double*, const double*, const double*);
    __global__ void k_viscous_fv(const K, const double*, const double*, const double*, double*,
                                 double*, double*, const double*, const double*, const double*,
                                 const double*, const double*, const double*);
    __global__ void k_viscous_fw(const K, const double*, const double*, const double*, double*,
                                 double*, double*, const double*, const double*, const double*,
                                 const double*, const double*, const double*);
    __global__ void k_w1_w2_dx(const K, const double*, const double*, double*, double*,
                               const double*);
    __global__ void k_w3_dx(const K, const double*, double*, const double*);
    __global__ void k_ft_heat1(const K, const double*, const double*, double*, const double*,
                               const double*, const double*);
    __global__ void k_cross_fufv(const K, double*, double*, const double*, const double*,
                                 const double*);
    __global__ void k_w1_dy(const K, const double*, double*, const double*);
    __global__ void k_w2_dy(const K, const double*, double*, const double*);
    __global__ void k_ft_heat2a(const K, const double*, double*, const double*, const double*);
    __global__ void k_ft_heat2b(const K, const double*, const double*, double*, const double*);
    __global__ void k_w1_dx_again(const K, const double*, double*, const double*);
    __global__ void k_w3_dz_full(const K, const double*, double*, const double*);
    __global__ void k_ft_heat3a(const K, const double*, const double*, double*, const double*,
                                const double*, const double*);
    __global__ void k_cross_w3(const K, double*, double*, const double*, const double*,
                               const double*);
    __global__ void k_w1_w2_dz(const K, const double*, const double*, double*, double*,
                               const double*);
    __global__ void k_w3_dy(const K, const double*, double*, const double*);
    __global__ void k_ft_heat3c(const K, const double*, double*, const double*, const double*,
                                const double*);
    __global__ void k_fw_cross(const K, double*, const double*, const double*, const double*,
                               const double*);
    __global__ void k_w3_dx_2(const K, const double*, double*, const double*);
    __global__ void k_ft_heat3d(const K, const double*, double*, const double*, const double*);
    __global__ void k_rot(const K, double*, double*, double*, const double*, const double*,
                          const double*);
    __global__ void k_step_uu(const K, const double*, const double*, const double*, const double*,
                              const double*, double*);
    __global__ void k_step_dt(const K, const double*, const double*, const double*, const double*,
                              const double*, const double*, double*, double*, double*);
    __global__ void k_dt_min3(const K, const double*, const double*, const double*, double*);
    __global__ void k_dt_final(double*, const double*, int);
    __global__ void k_save_state(const K, const double*, const double*, const double*,
                                 const double*, const double*, double*, double*, double*, double*,
                                 double*);
    __global__ void k_update_prim(const K, double, const double*, const double*, const double*,
                                  const double*, const double*, const double*, const double*,
                                  const double*, const double*, const double*, double*, double*,
                                  double*, double*, double*);
    __global__ void k_mix_state(const K, double, const double*, const double*, const double*,
                                const double*, const double*, const double*, const double*,
                                const double*, const double*, const double*, double*, double*,
                                double*, double*, double*);
    __global__ void k_zero_fill(double*, long);
    __global__ void k_sum(double*, long, double*);
    __global__ void k_bcon(const K, const double*, const double*, const double*, const double*,
                           double*, double*, double*, double*, double*, double, double, double,
                           double, double, double, double, double, int, int, int, int);
    __global__ void k_pack_y(double*, const double*, int, int, int, int, int);
    __global__ void k_unpack_y(double*, const double*, int, int, int, int, int);
    __global__ void k_pack_x(double*, const double*, int, int, int, int, int);
    __global__ void k_unpack_x(double*, const double*, int, int, int, int, int);
    // Device min/max reductions (atomic via bit-trick on __int64).

    namespace {

#define CUDA_CHECK(x)                                                                      \
    do {                                                                                   \
        cudaError_t e = (x);                                                               \
        if (e != cudaSuccess) {                                                            \
            fprintf(stderr, "rast-3dmhd: CUDA error %s at %s:%d\n", cudaGetErrorString(e), \
                    __FILE__, __LINE__);                                                   \
            fflush(stderr); /* never lose the message before exit() */                     \
            exit(EXIT_FAILURE);                                                            \
        }                                                                                  \
    } while (0)

#ifdef R3D_COMM_NCCL
#define NCCL_CHECK(x)                                                                      \
    do {                                                                                   \
        ncclResult_t e = (x);                                                              \
        if (e != ncclSuccess) {                                                            \
            fprintf(stderr, "rast-3dmhd: NCCL error %s at %s:%d\n", ncclGetErrorString(e), \
                    __FILE__, __LINE__);                                                   \
            fflush(stderr);                                                                \
            exit(EXIT_FAILURE);                                                            \
        }                                                                                  \
    } while (0)
#endif

        constexpr int kThreads = 256;

        long ceil_div(long n, int t) { return (n + t - 1) / t; }

        void alloc(double** p, long n) {
            CUDA_CHECK(cudaMalloc((void**) p, (size_t) n * sizeof(double)));
        }
        void zero(double* p, long n) { CUDA_CHECK(cudaGetLastError()); }
        void check() { CUDA_CHECK(cudaGetLastError()); }

#ifdef R3D_COMM_NCCL
        // One NCCL communicator over the (single) MPI communicator.  Created
        // lazily on the first step_gpu call - all ranks reach it in lockstep,
        // which the collective unique-id broadcast needs - and destroyed from
        // gpu_free() so the tests can re-init cleanly between cases.
        ncclComm_t g_nccl_comm = nullptr;
        bool g_nccl_ready      = false;

        void nccl_ensure(MPI_Comm comm) {
            if (g_nccl_ready) return;
            int npe, mype;
            MPI_Comm_size(comm, &npe);
            MPI_Comm_rank(comm, &mype);
            ncclUniqueId id;
            if (mype == 0) NCCL_CHECK(ncclGetUniqueId(&id));
            MPI_Bcast(&id, (int) sizeof(id), MPI_BYTE, 0, comm);
            NCCL_CHECK(ncclCommInitRank(&g_nccl_comm, npe, id, mype));
            g_nccl_ready = true;
        }

        void nccl_teardown() {
            if (g_nccl_ready) {
                NCCL_CHECK(ncclCommDestroy(g_nccl_comm));
                g_nccl_comm  = nullptr;
                g_nccl_ready = false;
            }
        }
#endif

    }  // namespace

    // ---------------------------------------------------------------------------
    // Allocation / transfer.
    // ---------------------------------------------------------------------------
    void gpu_alloc(GpuSim& g, const Derived& d) {
        g.nx   = d.nx;
        g.ny   = d.ny;
        g.nz   = d.nz;
        long n = (long) d.nx * d.ny * d.nz;
        alloc(&g.ru, n);
        alloc(&g.rv, n);
        alloc(&g.rw, n);
        alloc(&g.ro, n);
        alloc(&g.tt, n);
        alloc(&g.uu, n);
        alloc(&g.vv, n);
        alloc(&g.ww, n);
        alloc(&g.fu, n);
        alloc(&g.fv, n);
        alloc(&g.fw, n);
        alloc(&g.fr, n);
        alloc(&g.ft, n);
        alloc(&g.zru, n);
        alloc(&g.zrv, n);
        alloc(&g.zrw, n);
        alloc(&g.zro, n);
        alloc(&g.ztt, n);
        alloc(&g.ww1, n);
        alloc(&g.ww2, n);
        alloc(&g.ww3, n);
        alloc(&g.bx, n);
        alloc(&g.by, n);
        alloc(&g.bz, n);
        alloc(&g.zbx, n);
        alloc(&g.zby, n);
        alloc(&g.zbz, n);
        alloc(&g.exx, d.nx);
        alloc(&g.dxxdx, d.nx);
        alloc(&g.d2xxdx2, d.nx);
        alloc(&g.ddx, d.nx);
        alloc(&g.wyy, d.ny);
        alloc(&g.dyydy, d.ny);
        alloc(&g.d2yydy2, d.ny);
        alloc(&g.ddy, d.ny);
        alloc(&g.zee, d.nz);
        alloc(&g.dzzdz, d.nz);
        alloc(&g.d2zzdz2, d.nz);
        alloc(&g.ddz, d.nz);
        alloc(&g.rkapa, d.nz);
        alloc(&g.dkapa, d.nz);
        alloc(&g.rho0, n);  // pristine rho snapshot
    }

    void gpu_free(GpuSim& g) {
#define FREE(x)      \
    if (x) {         \
        cudaFree(x); \
        x = nullptr; \
    }
        FREE(g.ru);
        FREE(g.rv);
        FREE(g.rw);
        FREE(g.ro);
        FREE(g.tt);
        FREE(g.uu);
        FREE(g.vv);
        FREE(g.ww);
        FREE(g.fu);
        FREE(g.fv);
        FREE(g.fw);
        FREE(g.fr);
        FREE(g.ft);
        FREE(g.zru);
        FREE(g.zrv);
        FREE(g.zrw);
        FREE(g.zro);
        FREE(g.ztt);
        FREE(g.ww1);
        FREE(g.ww2);
        FREE(g.ww3);
        FREE(g.bx);
        FREE(g.by);
        FREE(g.bz);
        FREE(g.zbx);
        FREE(g.zby);
        FREE(g.zbz);
        FREE(g.exx);
        FREE(g.dxxdx);
        FREE(g.d2xxdx2);
        FREE(g.ddx);
        FREE(g.wyy);
        FREE(g.dyydy);
        FREE(g.d2yydy2);
        FREE(g.ddy);
        FREE(g.zee);
        FREE(g.dzzdz);
        FREE(g.d2zzdz2);
        FREE(g.ddz);
        FREE(g.rkapa);
        FREE(g.dkapa);
        FREE(g.rho0);
#undef FREE
#ifdef R3D_COMM_NCCL
        nccl_teardown();
#endif
    }

    static void cp(double* dst, const double* src, long n) {
        CUDA_CHECK(cudaMemcpy(dst, src, (size_t) n * sizeof(double), cudaMemcpyHostToDevice));
    }
    static void cp_back(double* dst, const double* src, long n) {
        CUDA_CHECK(cudaMemcpy(dst, src, (size_t) n * sizeof(double), cudaMemcpyDeviceToHost));
    }

    void gpu_upload(GpuSim& g, const SimState& s) {
        long n = (long) g.nx * g.ny * g.nz;
        cp(g.ru, s.ru.data(), n);
        cp(g.rv, s.rv.data(), n);
        cp(g.rw, s.rw.data(), n);
        cp(g.ro, s.ro.data(), n);
        cp(g.tt, s.tt.data(), n);
        cp(g.uu, s.uu.data(), n);
        cp(g.vv, s.vv.data(), n);
        cp(g.ww, s.ww.data(), n);
        cp(g.fu, s.fu.data(), n);
        cp(g.fv, s.fv.data(), n);
        cp(g.fw, s.fw.data(), n);
        cp(g.fr, s.fr.data(), n);
        cp(g.ft, s.ft.data(), n);
        cp(g.zru, s.zru.data(), n);
        cp(g.zrv, s.zrv.data(), n);
        cp(g.zrw, s.zrw.data(), n);
        cp(g.zro, s.zro.data(), n);
        cp(g.ztt, s.ztt.data(), n);
        cp(g.ww1, s.ww1.data(), n);
        cp(g.ww2, s.ww2.data(), n);
        cp(g.ww3, s.ww3.data(), n);
        cp(g.bx, s.bx.data(), n);
        cp(g.by, s.by.data(), n);
        cp(g.bz, s.bz.data(), n);
        cp(g.zbx, s.zbx.data(), n);
        cp(g.zby, s.zby.data(), n);
        cp(g.zbz, s.zbz.data(), n);
        cp(g.exx, s.exx.data(), g.nx);
        cp(g.dxxdx, s.dxxdx.data(), g.nx);
        cp(g.d2xxdx2, s.d2xxdx2.data(), g.nx);
        cp(g.ddx, s.ddx.data(), g.nx);
        cp(g.wyy, s.wyy.data(), g.ny);
        cp(g.dyydy, s.dyydy.data(), g.ny);
        cp(g.d2yydy2, s.d2yydy2.data(), g.ny);
        cp(g.ddy, s.ddy.data(), g.ny);
        cp(g.zee, s.zee.data(), g.nz);
        cp(g.dzzdz, s.dzzdz.data(), g.nz);
        cp(g.d2zzdz2, s.d2zzdz2.data(), g.nz);
        cp(g.ddz, s.ddz.data(), g.nz);
        cp(g.rkapa, s.rkapa.data(), g.nz);
        cp(g.dkapa, s.dkapa.data(), g.nz);
        cp(g.rho0, s.ro.data(), n);  // initial pristine rho
    }

    void gpu_download(SimState& s, GpuSim& g) {
        long n = (long) g.nx * g.ny * g.nz;
        cp_back(s.ru.data(), g.ru, n);
        cp_back(s.rv.data(), g.rv, n);
        cp_back(s.rw.data(), g.rw, n);
        cp_back(s.ro.data(), g.ro, n);
        cp_back(s.tt.data(), g.tt, n);
        cp_back(s.uu.data(), g.uu, n);
        cp_back(s.vv.data(), g.vv, n);
        cp_back(s.ww.data(), g.ww, n);
        cp_back(s.fu.data(), g.fu, n);
        cp_back(s.fv.data(), g.fv, n);
        cp_back(s.fw.data(), g.fw, n);
        cp_back(s.fr.data(), g.fr, n);
        cp_back(s.ft.data(), g.ft, n);
        cp_back(s.zru.data(), g.zru, n);
        cp_back(s.zrv.data(), g.zrv, n);
        cp_back(s.zrw.data(), g.zrw, n);
        cp_back(s.zro.data(), g.zro, n);
        cp_back(s.ztt.data(), g.ztt, n);
        cp_back(s.ww1.data(), g.ww1, n);
        cp_back(s.ww2.data(), g.ww2, n);
        cp_back(s.ww3.data(), g.ww3, n);
        cp_back(s.bx.data(), g.bx, n);
        cp_back(s.by.data(), g.by, n);
        cp_back(s.bz.data(), g.bz, n);
        cp_back(s.zbx.data(), g.zbx, n);
        cp_back(s.zby.data(), g.zby, n);
        cp_back(s.zbz.data(), g.zbz, n);
        cp_back(s.exx.data(), g.exx, g.nx);
        cp_back(s.dxxdx.data(), g.dxxdx, g.nx);
        cp_back(s.d2xxdx2.data(), g.d2xxdx2, g.nx);
        cp_back(s.ddx.data(), g.ddx, g.nx);
        cp_back(s.wyy.data(), g.wyy, g.ny);
        cp_back(s.dyydy.data(), g.dyydy, g.ny);
        cp_back(s.d2yydy2.data(), g.d2yydy2, g.ny);
        cp_back(s.ddy.data(), g.ddy, g.ny);
        cp_back(s.zee.data(), g.zee, g.nz);
        cp_back(s.dzzdz.data(), g.dzzdz, g.nz);
        cp_back(s.d2zzdz2.data(), g.d2zzdz2, g.nz);
        cp_back(s.ddz.data(), g.ddz, g.nz);
        cp_back(s.rkapa.data(), g.rkapa, g.nz);
        cp_back(s.dkapa.data(), g.dkapa, g.nz);
    }

    // ---------------------------------------------------------------------------
    // Device halo exchange (CUDA-aware MPI + pack kernels).
    // ---------------------------------------------------------------------------
    namespace {

        void comm_field_gpu(MPI_Comm comm, const Params& p, const Topology& t, GpuSim& g,
                            double* f) {
            // Host-staged halo exchange.  Every MPI transfer touches only pinned
            // HOST memory (ordinary MPI semantics, reliable on all transports);
            // device<->host transfers are cudaMemcpyAsync on the default stream,
            // so ordering with the pack/unpack kernels on that stream is
            // deterministic by construction.  (Passing device pointers to CUDA-aware
            // MPI instead proved unreliable here: UCX performs its copies on an
            // internal stream that a blocking Recv return and even a default-stream
            // sync do not join, so consumer kernels could read a previous message.
            // GPU-direct halo exchange via NCCL is the planned follow-up.)
            const int nx = g.nx, ny = g.ny, nz = g.nz;
            const int iyh = p.iy / 2, ilaph = p.ilap / 2;
            const long vcount   = (long) nx * ny * ilaph;
            const long ycount   = (long) nx * iyh * nz;
            static double* hbuf = nullptr;  // pinned host staging (send + recv)
            static long hcap    = 0;
            const long hneed    = std::max(vcount, ycount);
            if (hcap < hneed) {
                if (hbuf) cudaFreeHost(hbuf);
                CUDA_CHECK(cudaMallocHost((void**) &hbuf, (size_t) hneed * sizeof(double)));
                hcap = hneed;
            }
            // Blocking copies: the transfers are tiny (halo slabs) and blocking
            // copies make every ordering between hbuf and the default-stream
            // kernels deterministic by construction (no async-use-after-reuse).
            auto copy_to_dev = [&](double* dst, long cnt) {
                CUDA_CHECK(
                    cudaMemcpy(dst, hbuf, (size_t) cnt * sizeof(double), cudaMemcpyHostToDevice));
            };
            auto copy_to_host = [&](const double* src, long cnt) {
                CUDA_CHECK(
                    cudaMemcpy(hbuf, src, (size_t) cnt * sizeof(double), cudaMemcpyDeviceToHost));
            };
            const double* vbot = f + (long) (nz - p.ilap) * nx * ny;  // send (bottom)
            double* vghost     = f + (long) (nz - ilaph) * nx * ny;   // recv (upper)
            // Vertical z exchange (host-staged on BOTH sides - the slab is
            // written by default-stream kernels; a direct device send could be
            // consumed by the transport's internal stream before those kernels
            // complete, the same ordering hazard we hit on receives).
            if (t.npez > 1) {
                const int tag1 = 100, tag2 = 200;
                if (t.mypez == 0) {
                    copy_to_host(vbot, vcount);
                    MPI_Send(hbuf, vcount, MPI_DOUBLE, t.mype + t.npey, tag1, comm);
                    MPI_Recv(hbuf, vcount, MPI_DOUBLE, t.mype + t.npey, tag2, comm,
                             MPI_STATUS_IGNORE);
                    copy_to_dev(vghost, vcount);
                } else if (t.mypez == t.npez - 1) {
                    MPI_Recv(hbuf, vcount, MPI_DOUBLE, t.mype - t.npey, tag1, comm,
                             MPI_STATUS_IGNORE);
                    copy_to_dev(f, vcount);
                    copy_to_host(f + (long) ilaph * nx * ny, vcount);
                    MPI_Send(hbuf, vcount, MPI_DOUBLE, t.mype - t.npey, tag2, comm);
                } else {
                    copy_to_host(vbot, vcount);
                    MPI_Sendrecv(hbuf, vcount, MPI_DOUBLE, t.mype + t.npey, tag1, hbuf, vcount,
                                 MPI_DOUBLE, t.mype - t.npey, tag1, comm, MPI_STATUS_IGNORE);
                    copy_to_dev(f, vcount);
                    copy_to_host(f + (long) ilaph * nx * ny, vcount);
                    MPI_Sendrecv(hbuf, vcount, MPI_DOUBLE, t.mype - t.npey, tag2, hbuf, vcount,
                                 MPI_DOUBLE, t.mype + t.npey, tag2, comm, MPI_STATUS_IGNORE);
                    copy_to_dev(vghost, vcount);
                }
            }
            // device staging buffers for the strided y/x exchanges
            static double* ysend = nullptr;
            static double* yrecv = nullptr;
            static long ycap     = 0;
            if (ycap < ycount) {
                if (ysend) {
                    cudaFree(ysend);
                    cudaFree(yrecv);
                }
                CUDA_CHECK(cudaMalloc((void**) &ysend, (size_t) ycount * sizeof(double)));
                CUDA_CHECK(cudaMalloc((void**) &yrecv, (size_t) ycount * sizeof(double)));
                ycap = ycount;
            }
            if (t.npey > 1) {
                const int tag3 = 300, tag4 = 400, tag5 = 500, tag6 = 600;
                if (t.mypey == 0) {
                    k_pack_y<<<ceil_div(ycount, kThreads), kThreads>>>(ysend, f, nx, ny, ny - p.iy,
                                                                       iyh, nz);
                    check();
                    copy_to_host(ysend, ycount);
                    MPI_Send(hbuf, ycount, MPI_DOUBLE, t.mype + 1, tag3, comm);
                    MPI_Recv(hbuf, ycount, MPI_DOUBLE, t.mype + 1, tag4, comm, MPI_STATUS_IGNORE);
                    copy_to_dev(yrecv, ycount);
                    k_unpack_y<<<ceil_div(ycount, kThreads), kThreads>>>(f, yrecv, nx, ny, ny - iyh,
                                                                         iyh, nz);
                    check();
                } else if (t.mypey == t.npey - 1) {
                    MPI_Recv(hbuf, ycount, MPI_DOUBLE, t.mype - 1, tag3, comm, MPI_STATUS_IGNORE);
                    copy_to_dev(yrecv, ycount);
                    k_unpack_y<<<ceil_div(ycount, kThreads), kThreads>>>(f, yrecv, nx, ny, 0, iyh,
                                                                         nz);
                    check();
                    k_pack_y<<<ceil_div(ycount, kThreads), kThreads>>>(ysend, f, nx, ny, iyh, iyh,
                                                                       nz);
                    check();
                    copy_to_host(ysend, ycount);
                    MPI_Send(hbuf, ycount, MPI_DOUBLE, t.mype - 1, tag4, comm);
                } else {
                    k_pack_y<<<ceil_div(ycount, kThreads), kThreads>>>(ysend, f, nx, ny, ny - p.iy,
                                                                       iyh, nz);
                    check();
                    copy_to_host(ysend, ycount);
                    MPI_Sendrecv(hbuf, ycount, MPI_DOUBLE, t.mype + 1, tag3, hbuf, ycount,
                                 MPI_DOUBLE, t.mype - 1, tag3, comm, MPI_STATUS_IGNORE);
                    copy_to_dev(yrecv, ycount);
                    k_unpack_y<<<ceil_div(ycount, kThreads), kThreads>>>(f, yrecv, nx, ny, 0, iyh,
                                                                         nz);
                    check();
                    k_pack_y<<<ceil_div(ycount, kThreads), kThreads>>>(ysend, f, nx, ny, iyh, iyh,
                                                                       nz);
                    check();
                    copy_to_host(ysend, ycount);
                    MPI_Sendrecv(hbuf, ycount, MPI_DOUBLE, t.mype - 1, tag4, hbuf, ycount,
                                 MPI_DOUBLE, t.mype + 1, tag4, comm, MPI_STATUS_IGNORE);
                    copy_to_dev(yrecv, ycount);
                    k_unpack_y<<<ceil_div(ycount, kThreads), kThreads>>>(f, yrecv, nx, ny, ny - iyh,
                                                                         iyh, nz);
                    check();
                }
                if (t.mypey == 0) {
                    k_pack_y<<<ceil_div(ycount, kThreads), kThreads>>>(ysend, f, nx, ny, iyh, iyh,
                                                                       nz);
                    check();
                    copy_to_host(ysend, ycount);
                    MPI_Send(hbuf, ycount, MPI_DOUBLE, t.mype + t.npey - 1, tag5, comm);
                    MPI_Recv(hbuf, ycount, MPI_DOUBLE, t.mype + t.npey - 1, tag6, comm,
                             MPI_STATUS_IGNORE);
                    copy_to_dev(yrecv, ycount);
                    k_unpack_y<<<ceil_div(ycount, kThreads), kThreads>>>(f, yrecv, nx, ny, 0, iyh,
                                                                         nz);
                    check();
                } else if (t.mypey == t.npey - 1) {
                    MPI_Recv(hbuf, ycount, MPI_DOUBLE, t.mype - t.npey + 1, tag5, comm,
                             MPI_STATUS_IGNORE);
                    copy_to_dev(yrecv, ycount);
                    k_unpack_y<<<ceil_div(ycount, kThreads), kThreads>>>(f, yrecv, nx, ny, ny - iyh,
                                                                         iyh, nz);
                    check();
                    k_pack_y<<<ceil_div(ycount, kThreads), kThreads>>>(ysend, f, nx, ny, ny - p.iy,
                                                                       iyh, nz);
                    check();
                    copy_to_host(ysend, ycount);
                    MPI_Send(hbuf, ycount, MPI_DOUBLE, t.mype - t.npey + 1, tag6, comm);
                }
            } else {
                k_pack_y<<<ceil_div(ycount, kThreads), kThreads>>>(ysend, f, nx, ny, ny - p.iy, iyh,
                                                                   nz);
                check();
                k_unpack_y<<<ceil_div(ycount, kThreads), kThreads>>>(f, ysend, nx, ny, 0, iyh, nz);
                check();
                k_pack_y<<<ceil_div(ycount, kThreads), kThreads>>>(ysend, f, nx, ny, iyh, iyh, nz);
                check();
                k_unpack_y<<<ceil_div(ycount, kThreads), kThreads>>>(f, ysend, nx, ny, ny - iyh,
                                                                     iyh, nz);
                check();
            }
            const long xcount   = (long) (p.ix / 2) * ny * nz;
            static double* xbuf = nullptr;
            static long xcap    = 0;
            if (xcap < xcount) {
                if (xbuf) cudaFree(xbuf);
                CUDA_CHECK(cudaMalloc((void**) &xbuf, (size_t) xcount * sizeof(double)));
                xcap = xcount;
            }
            k_pack_x<<<ceil_div(xcount, kThreads), kThreads>>>(xbuf, f, nx, ny, nx - p.ix, p.ix / 2,
                                                               nz);
            check();
            k_unpack_x<<<ceil_div(xcount, kThreads), kThreads>>>(f, xbuf, nx, ny, 0, p.ix / 2, nz);
            check();
            k_pack_x<<<ceil_div(xcount, kThreads), kThreads>>>(xbuf, f, nx, ny, p.ix / 2, p.ix / 2,
                                                               nz);
            check();
            k_unpack_x<<<ceil_div(xcount, kThreads), kThreads>>>(f, xbuf, nx, ny, nx - p.ix / 2,
                                                                 p.ix / 2, nz);
            check();
        }
    }  // namespace

    // ---------------------------------------------------------------------------
    // NCCL device-side communication (compiled only with -DR3D_COMM_NCCL).
    //
    // The halo exchange and the per-step dt reduction run entirely on the
    // device: strided y/z slabs are packed once into device staging buffers and
    // moved straight to the neighbour's GPU (no pinned-host bounce, no
    // transport-dependent CUDA-aware MPI memory-ordering hazards).  NCCL
    // operations are issued on the default stream, so they are ordered with the
    // pack/unpack kernels by construction and (being GPU-blocking) complete
    // before the next same-stream kernel runs - staging buffers are safe to
    // reuse across fields without an explicit sync.
    // ---------------------------------------------------------------------------
#ifdef R3D_COMM_NCCL
    namespace {

        // Device halo exchange for one field.  Equivalent to comm_field_gpu()
        // above but with device-side NCCL collectives instead of the host-staged
        // MPI bounce.  Each axis is one ncclAllGather: every rank contributes its
        // interior slabs (for z: the send-up and send-down planes; for y: the
        // packed upper and lower slabs) and then unpacks the two neighbour slabs
        // it needs from the gathered record.  AllGather uses the ring/tree
        // collective machinery - unlike NCCL's point-to-point send/recv pipe,
        // which fails to come up on some clusters/GPUs - so this is the portable
        // NCCL path.  The values moved are byte-identical to the MPI path's.
        void comm_field_gpu_nccl(const Params& p, const Topology& t, GpuSim& g, double* f) {
            const int nx = g.nx, ny = g.ny, nz = g.nz;
            const int iyh = p.iy / 2, ilaph = p.ilap / 2;
            const long vcount = (long) nx * ny * ilaph;
            const long ycount = (long) nx * iyh * nz;
            const int nranks  = t.npe;
            cudaStream_t s    = nullptr;  // default stream: ordered with the kernels

            // Vertical (z) halo: one AllGather of the two interior slabs (zslot
            // = [send-up, send-down]).  Lower ghost comes from the rank below's
            // send-up slab, upper ghost from the rank above's send-down slab.
            // npez==1 (or the boundary end of the line) simply has no such
            // neighbour.  The gather runs over the whole comm - slabs contributed
            // by ranks in other z-columns are unused, which is harmless.
            if (t.npez > 1) {
                const long zslot     = 2 * vcount;
                static double *zsend = nullptr, *zgather = nullptr;
                static long zcap = 0;
                if (zcap < (long) nranks * zslot) {
                    if (zsend) {
                        cudaFree(zsend);
                        cudaFree(zgather);
                    }
                    CUDA_CHECK(cudaMalloc((void**) &zsend, (size_t) zslot * sizeof(double)));
                    CUDA_CHECK(
                        cudaMalloc((void**) &zgather, (size_t) nranks * zslot * sizeof(double)));
                    zcap = (long) nranks * zslot;
                }
                CUDA_CHECK(cudaMemcpyAsync(zsend, f + (long) (nz - p.ilap) * nx * ny,
                                           (size_t) vcount * sizeof(double),
                                           cudaMemcpyDeviceToDevice, s));
                CUDA_CHECK(cudaMemcpyAsync(zsend + vcount, f + (long) ilaph * nx * ny,
                                           (size_t) vcount * sizeof(double),
                                           cudaMemcpyDeviceToDevice, s));
                NCCL_CHECK(ncclAllGather(zsend, zgather, zslot, ncclDouble, g_nccl_comm, s));
                if (t.mypez > 0)  // lower ghost plane from the rank below
                    CUDA_CHECK(cudaMemcpyAsync(f, zgather + (long) (t.mype - t.npey) * zslot,
                                               (size_t) vcount * sizeof(double),
                                               cudaMemcpyDeviceToDevice, s));
                if (t.mypez < t.npez - 1)  // upper ghost plane from the rank above
                    CUDA_CHECK(cudaMemcpyAsync(f + (long) (nz - ilaph) * nx * ny,
                                               zgather + (long) (t.mype + t.npey) * zslot + vcount,
                                               (size_t) vcount * sizeof(double),
                                               cudaMemcpyDeviceToDevice, s));
            }

            // y halo: one AllGather of the two strided slabs, pre-packed into a
            // device buffer (yslot = [upper, lower]).  The low ghost is the rank
            // below's upper slab, the high ghost the rank above's lower slab;
            // neighbours are the cyclic (mypey +/- 1) columns, which for npey==1
            // means this rank wraps onto itself (the MPI path's local branch).
            // The per-rank record lives at the *rank* slot (rank = mypey + mypez
            // * npey), so neighbours are addressed by rank, not mypey.
            {
                const long yslot     = 2 * ycount;
                static double *ysend = nullptr, *ygather = nullptr;
                static long ycap = 0;
                if (ycap < (long) nranks * yslot) {
                    if (ysend) {
                        cudaFree(ysend);
                        cudaFree(ygather);
                    }
                    CUDA_CHECK(cudaMalloc((void**) &ysend, (size_t) yslot * sizeof(double)));
                    CUDA_CHECK(
                        cudaMalloc((void**) &ygather, (size_t) nranks * yslot * sizeof(double)));
                    ycap = (long) nranks * yslot;
                }
                const int rDn = t.mype - t.mypey + (t.mypey + t.npey - 1) % t.npey;
                const int rUp = t.mype - t.mypey + (t.mypey + 1) % t.npey;
                k_pack_y<<<ceil_div(ycount, kThreads), kThreads>>>(ysend, f, nx, ny, ny - p.iy, iyh,
                                                                   nz);
                k_pack_y<<<ceil_div(ycount, kThreads), kThreads>>>(ysend + ycount, f, nx, ny, iyh,
                                                                   iyh, nz);
                check();
                NCCL_CHECK(ncclAllGather(ysend, ygather, yslot, ncclDouble, g_nccl_comm, s));
                k_unpack_y<<<ceil_div(ycount, kThreads), kThreads>>>(
                    f, ygather + (long) rDn * yslot, nx, ny, 0, iyh, nz);
                k_unpack_y<<<ceil_div(ycount, kThreads), kThreads>>>(
                    f, ygather + (long) rUp * yslot + ycount, nx, ny, ny - iyh, iyh, nz);
                check();
            }

            // x halo: x is never decomposed, so this is a purely local periodic
            // wrap through a single device staging buffer (unchanged from the MPI
            // path).
            const long xcount   = (long) (p.ix / 2) * ny * nz;
            static double* xbuf = nullptr;
            static long xcap    = 0;
            if (xcap < xcount) {
                if (xbuf) cudaFree(xbuf);
                CUDA_CHECK(cudaMalloc((void**) &xbuf, (size_t) xcount * sizeof(double)));
                xcap = xcount;
            }
            k_pack_x<<<ceil_div(xcount, kThreads), kThreads>>>(xbuf, f, nx, ny, nx - p.ix, p.ix / 2,
                                                               nz);
            check();
            k_unpack_x<<<ceil_div(xcount, kThreads), kThreads>>>(f, xbuf, nx, ny, 0, p.ix / 2, nz);
            check();
            k_pack_x<<<ceil_div(xcount, kThreads), kThreads>>>(xbuf, f, nx, ny, p.ix / 2, p.ix / 2,
                                                               nz);
            check();
            k_unpack_x<<<ceil_div(xcount, kThreads), kThreads>>>(f, xbuf, nx, ny, nx - p.ix / 2,
                                                                 p.ix / 2, nz);
            check();
        }

        // Per-step timestep: interior min of WW1..3 on the device, then one NCCL
        // AllReduce (min) over the ranks.  Keeps the whole reduction on the GPU -
        // the MPI path's three full-array D2H copies per step are gone.
        void step_dt_gpu_nccl(const Params& p, const K& q, GpuSim& g, double& dt) {
            const long interior = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
            const int nblocks   = (int) ceil_div(interior, kThreads);
            static double* partial = nullptr;
            static long pcap       = 0;
            if (pcap < 3 * (long) nblocks) {
                if (partial) cudaFree(partial);
                CUDA_CHECK(cudaMalloc((void**) &partial, 3 * (size_t) nblocks * sizeof(double)));
                pcap = 3 * (long) nblocks;
            }
            static double* out = nullptr;
            if (!out) CUDA_CHECK(cudaMalloc((void**) &out, 3 * sizeof(double)));
            cudaStream_t s = nullptr;
            k_dt_min3<<<nblocks, kThreads>>>(q, g.ww1, g.ww2, g.ww3, partial);
            check();
            k_dt_final<<<1, kThreads>>>(out, partial, nblocks);
            check();
            NCCL_CHECK(ncclAllReduce(out, out, 3, ncclDouble, ncclMin, g_nccl_comm, s));
            static double hout[3];
            CUDA_CHECK(cudaMemcpy(hout, out, 3 * sizeof(double), cudaMemcpyDeviceToHost));
            dt = p.sf * std::min(std::min(hout[0], hout[1]), hout[2]);
        }

    }  // namespace
#endif

    // ---------------------------------------------------------------------------
    // step_gpu
    // ---------------------------------------------------------------------------
    void step_gpu(int nstages, MPI_Comm comm, const Params& p, const Derived& d, const Topology& t,
                  GpuSim& g, SimState& sc) {
#ifdef R3D_COMM_NCCL
        nccl_ensure(comm);
#endif
        K q;
        q.nx    = g.nx;
        q.ny    = g.ny;
        q.nz    = g.nz;
        q.i1    = 1;
        q.i2    = g.nx - p.ix;
        q.j1    = 1;
        q.j2    = g.ny - p.iy;
        q.k1    = p.ilap / 2;
        q.k2    = g.nz - 1 - p.ilap / 2;
        q.c13   = d.c13;
        q.c23   = d.c23;
        q.c43   = d.c43;
        q.ore   = d.ore;
        q.ocv   = d.ocv;
        q.repr  = d.repr;
        q.grav  = p.grav;
        q.hx    = d.hx;
        q.h2x   = d.h2x;
        q.hy    = d.hy;
        q.h2y   = d.h2y;
        q.hz    = d.hz;
        q.h2z   = d.h2z;
        q.omx   = d.omx;
        q.omz   = d.omz;
        q.gamma = p.gamma;
        q.re    = p.re;
        q.cv    = d.cv;
        q.xp    = p.xp;
        q.yp    = p.yp;
        q.hh    = p.hh;
        q.li    = p.ilap / 2;
        q.lj    = p.iy / 2;
        q.lk    = p.ix / 2;

        const long n = (long) g.nx * g.ny * g.nz;
#ifdef R3D_COMM_NCCL
        // NCCL path: the whole dt reduction stays on the device - k_step_dt fills
        // WW1..3, a device interior-min folds them, and one NCCL AllReduce joins
        // the ranks (no host round-trip, no per-step D2H of the temporaries).
        k_step_dt<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                             kThreads),
                    kThreads>>>(q, g.ro, g.uu, g.rkapa, g.ddx, g.ddy, g.ddz, g.ww1, g.ww2, g.ww3);
        check();
        step_dt_gpu_nccl(p, q, g, sc.dt);
#else
        // For correctness and simplicity the few per-step reduction scalars are
        // computed on the host from a device -> host copy of the temporary arrays.
        // (They are single doubles; FLUXES stays fully on the device.)
        static double* host_tmp = nullptr;
        if (!host_tmp) host_tmp = (double*) malloc(n * sizeof(double));
        CUDA_CHECK(cudaMemcpy(host_tmp, g.uu, (size_t) n * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy((void*) host_tmp, g.uu, sizeof(double), cudaMemcpyDeviceToHost));
        // Use k_step_dt and a host min over the (small, test) geometry of ww1..3:
        k_step_dt<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                             kThreads),
                    kThreads>>>(q, g.ro, g.uu, g.rkapa, g.ddx, g.ddy, g.ddz, g.ww1, g.ww2, g.ww3);
        check();
        // wmin reductions done on host for the small test geometry; documented as
        // the single D2H of three arrays per step.
        // The per-cell timestep limits are only meaningful on the interior; the
        // ghost cells they are undefined on must not enter the reduction.
        auto interior_min = [&](double* dev, double lo) {
            double m = lo;
            for (int k = q.k1; k <= q.k2; ++k)
                for (int j = q.j1; j <= q.j2; ++j)
                    for (int i = q.i1; i <= q.i2; ++i)
                        m = std::min(m, dev[(long) k * g.nx * g.ny + (long) j * g.nx + i]);
            return m;
        };
        CUDA_CHECK(
            cudaMemcpy(host_tmp, g.ww1, (size_t) n * sizeof(double), cudaMemcpyDeviceToHost));
        double wmin[4] = {1e300, 1e300, 1e300, 1e300};
        wmin[0]        = interior_min(host_tmp, 1e300);
        CUDA_CHECK(
            cudaMemcpy(host_tmp, g.ww2, (size_t) n * sizeof(double), cudaMemcpyDeviceToHost));
        wmin[1] = interior_min(host_tmp, 1e300);
        CUDA_CHECK(
            cudaMemcpy(host_tmp, g.ww3, (size_t) n * sizeof(double), cudaMemcpyDeviceToHost));
        wmin[2] = interior_min(host_tmp, 1e300);
        double wout[4];
        MPI_Allreduce(wmin, wout, 3, MPI_DOUBLE, MPI_MIN, comm);
        sc.dt = p.sf * std::min(std::min(wout[0], wout[1]), wout[2]);
#endif

        // ---- 3-stage low-storage Wray RK3 on the device ----------------------
        k_save_state<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                                kThreads),
                       kThreads>>>(q, g.ru, g.rv, g.rw, g.ro, g.tt, g.zru, g.zrv, g.zrw, g.zro,
                                   g.ztt);
        check();

        auto flux_gpu = [&]() {
            k_fr<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                            kThreads),
                   kThreads>>>(q, g.ru, g.rv, g.rw, g.fr, g.ww1, g.dxxdx, g.dyydy, g.dzzdz);
            if (t.mypez == 0)
                k_fr_w1b<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1), kThreads),
                           kThreads>>>(q, g.rw, g.ww1, g.dzzdz, 1);
            check();
            if (t.mypez == t.npez - 1)
                k_fr_w1b<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1), kThreads),
                           kThreads>>>(q, g.rw, g.ww1, g.dzzdz, 0);
            check();
            k_fr_subw1<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                                  kThreads),
                         kThreads>>>(q, g.fr, g.ww1);
            // Snapshot the pristine rho before the in-place inversion, then run
            // the full-array prep and the interior force as separate kernels.
            CUDA_CHECK(
                cudaMemcpy(g.rho0, g.ro, (size_t) n * sizeof(double), cudaMemcpyDeviceToDevice));
            k_force_prep<<<ceil_div(n, kThreads), kThreads>>>(q, g.rho0, g.tt, g.ro, g.ww1);
            check();
            k_force<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                               kThreads),
                      kThreads>>>(q, p.grav, g.rho0, g.fu, g.fv, g.fw, g.ww1, g.dxxdx, g.dyydy,
                                  g.dzzdz);
            k_vel<<<ceil_div(n, kThreads), kThreads>>>(q, g.ru, g.rv, g.rw, g.ro, g.uu, g.vv, g.ww);
            k_adv<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                             kThreads),
                    kThreads>>>(q, 0, g.ru, g.rv, g.rw, g.uu, g.vv, g.ww, g.fu, g.fv, g.fw, g.dxxdx,
                                g.dyydy, g.dzzdz);
            check();
            k_adv<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                             kThreads),
                    kThreads>>>(q, 1, g.ru, g.rv, g.rw, g.uu, g.vv, g.ww, g.fu, g.fv, g.fw, g.dxxdx,
                                g.dyydy, g.dzzdz);
            k_adv<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                             kThreads),
                    kThreads>>>(q, 2, g.ru, g.rv, g.rw, g.uu, g.vv, g.ww, g.fu, g.fv, g.fw, g.dxxdx,
                                g.dyydy, g.dzzdz);
            check();
            k_ft_advect<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                                   kThreads),
                          kThreads>>>(q, g.uu, g.vv, g.ww, g.tt, g.ft, g.dxxdx, g.dyydy, g.dzzdz);
            k_ft_diffuse<<<ceil_div(
                               (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                               kThreads),
                           kThreads>>>(q, g.ro, g.tt, g.ft, g.dxxdx, g.d2xxdx2, g.dyydy, g.d2yydy2,
                                       g.dzzdz, g.d2zzdz2, g.rkapa, g.dkapa);
            check();
            k_viscous_fu<<<ceil_div(
                               (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                               kThreads),
                           kThreads>>>(q, g.uu, g.vv, g.ww, g.fu, g.fv, g.fw, g.dxxdx, g.d2xxdx2,
                                       g.dyydy, g.d2yydy2, g.dzzdz, g.d2zzdz2);
            k_viscous_fv<<<ceil_div(
                               (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                               kThreads),
                           kThreads>>>(q, g.uu, g.vv, g.ww, g.fu, g.fv, g.fw, g.dxxdx, g.d2xxdx2,
                                       g.dyydy, g.d2yydy2, g.dzzdz, g.d2zzdz2);
            k_viscous_fw<<<ceil_div(
                               (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                               kThreads),
                           kThreads>>>(q, g.uu, g.vv, g.ww, g.fu, g.fv, g.fw, g.dxxdx, g.d2xxdx2,
                                       g.dyydy, g.d2yydy2, g.dzzdz, g.d2zzdz2);
            check();
            k_w1_w2_dx<<<ceil_div((long) (q.i2 - q.i1 + 1) * q.ny * (q.k2 - q.k1 + 1), kThreads),
                         kThreads>>>(q, g.uu, g.vv, g.ww1, g.ww2, g.dxxdx);
            k_w3_dx<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                               kThreads),
                      kThreads>>>(q, g.ww, g.ww3, g.dxxdx);
            k_ft_heat1<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                                  kThreads),
                         kThreads>>>(q, g.ro, g.tt, g.ft, g.ww1, g.ww2, g.ww3);
            check();
            k_cross_fufv<<<ceil_div(
                               (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                               kThreads),
                           kThreads>>>(q, g.fu, g.fv, g.ww1, g.ww2, g.dyydy);
            // Fortran order: WW1=dU/dy -> FT+=(W1^2+2 W1 W2)RO*TMP using the
            // still-valid WW2=dV/dx -> THEN WW2=dV/dy -> FT+=W2^2..., FT-=ocv W2 T.
            k_w1_dy<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                               kThreads),
                      kThreads>>>(q, g.uu, g.ww1, g.dyydy);
            k_ft_heat2a<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                                   kThreads),
                          kThreads>>>(q, g.ro, g.ft, g.ww1, g.ww2);
            k_w2_dy<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                               kThreads),
                      kThreads>>>(q, g.vv, g.ww2, g.dyydy);
            k_ft_heat2b<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                                   kThreads),
                          kThreads>>>(q, g.ro, g.tt, g.ft, g.ww2);
            check();
            k_w1_dx_again<<<ceil_div(
                                (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                                kThreads),
                            kThreads>>>(q, g.uu, g.ww1, g.dxxdx);
            k_w3_dz_full<<<ceil_div((long) q.nx * q.ny * (q.k2 - q.k1 + 1), kThreads), kThreads>>>(
                q, g.ww, g.ww3, g.dzzdz);
            k_ft_heat3a<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                                   kThreads),
                          kThreads>>>(q, g.ro, g.tt, g.ft, g.ww1, g.ww2, g.ww3);
            check();
            k_cross_w3<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                                  kThreads),
                         kThreads>>>(q, g.fu, g.fv, g.ww3, g.dxxdx, g.dyydy);
            k_w1_w2_dz<<<ceil_div((long) q.nx * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1) +
                                      (long) (q.i2 - q.i1 + 1) * q.ny * (q.k2 - q.k1 + 1),
                                  kThreads),
                         kThreads>>>(q, g.uu, g.vv, g.ww1, g.ww2, g.dzzdz);
            k_w3_dy<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                               kThreads),
                      kThreads>>>(q, g.ww, g.ww3, g.dyydy);
            k_ft_heat3c<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                                   kThreads),
                          kThreads>>>(q, g.ro, g.ft, g.ww1, g.ww2, g.ww3);
            check();
            k_fw_cross<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                                  kThreads),
                         kThreads>>>(q, g.fw, g.ww1, g.ww2, g.dxxdx, g.dyydy);
            k_w3_dx_2<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                                 kThreads),
                        kThreads>>>(q, g.ww, g.ww3, g.dxxdx);
            k_ft_heat3d<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                                   kThreads),
                          kThreads>>>(q, g.ro, g.ft, g.ww1, g.ww3);
            if (p.lrot)
                k_rot<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                                 kThreads),
                        kThreads>>>(q, g.fu, g.fv, g.fw, g.ru, g.rv, g.rw);
            check();
        };

        auto bcon_gpu = [&]() {
            const double timt = sc.timc + sc.timi;
            double tpr        = (p.tc != 0.0) ? p.tp + (1.0 - p.tp) * exp(-timt / p.tc) : p.tp;
            const double cln  = f4(-4.0f * logf(2.0f)) / (p.hh * p.hh);
            const double dz   = f4(1.0f / (float) (p.npz - 1));
            int plume         = (p.tp != 0.0) && (p.itcon == 0 || p.itcon == 1);
            if (t.mypez == 0)
                k_bcon<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1), kThreads),
                         kThreads>>>(q, g.exx, g.wyy, g.dzzdz, g.rkapa, g.ru, g.rv, g.rw, g.ro,
                                     g.tt, d.c13, d.c23, d.c43, p.theta, sc.tb, tpr, cln, dz, plume,
                                     p.itcon, p.izcon, 1);
            check();
            if (t.mypez == t.npez - 1)
                k_bcon<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1), kThreads),
                         kThreads>>>(q, g.exx, g.wyy, g.dzzdz, g.rkapa, g.ru, g.rv, g.rw, g.ro,
                                     g.tt, d.c13, d.c23, d.c43, p.theta, sc.tb, tpr, cln, dz, 0,
                                     p.itcon, p.izcon, 0);
            check();
        };

        auto substep_gpu = [&](double coef) {
            k_update_prim<<<ceil_div(
                                (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                                kThreads),
                            kThreads>>>(q, coef, g.zru, g.zrv, g.zrw, g.zro, g.ztt, g.fu, g.fv,
                                        g.fw, g.fr, g.ft, g.ru, g.rv, g.rw, g.ro, g.tt);
            check();
        };
        auto mix_gpu = [&](double coef) {
            k_mix_state<<<ceil_div((long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1),
                                   kThreads),
                          kThreads>>>(q, coef, g.ru, g.rv, g.rw, g.ro, g.tt, g.fu, g.fv, g.fw, g.fr,
                                      g.ft, g.zru, g.zrv, g.zrw, g.zro, g.ztt);
            check();
        };
        auto comm_gpu = [&]() {
#ifdef R3D_COMM_NCCL
            comm_field_gpu_nccl(p, t, g, g.ru);
            comm_field_gpu_nccl(p, t, g, g.rv);
            comm_field_gpu_nccl(p, t, g, g.rw);
            comm_field_gpu_nccl(p, t, g, g.tt);
            comm_field_gpu_nccl(p, t, g, g.ro);
            if (p.lmag) {
                comm_field_gpu_nccl(p, t, g, g.bx);
                comm_field_gpu_nccl(p, t, g, g.by);
                comm_field_gpu_nccl(p, t, g, g.bz);
            }
#else
            comm_field_gpu(comm, p, t, g, g.ru);
            comm_field_gpu(comm, p, t, g, g.rv);
            comm_field_gpu(comm, p, t, g, g.rw);
            comm_field_gpu(comm, p, t, g, g.tt);
            comm_field_gpu(comm, p, t, g, g.ro);
            if (p.lmag) {
                comm_field_gpu(comm, p, t, g, g.bx);
                comm_field_gpu(comm, p, t, g, g.by);
                comm_field_gpu(comm, p, t, g, g.bz);
            }
#endif
        };

        for (int st = 0; st < nstages; ++st) {
            if (st == 0) {
                flux_gpu();
                substep_gpu(d.gam1 * sc.dt);
            } else if (st == 1) {
                mix_gpu(d.zeta1 * sc.dt);
                flux_gpu();
                substep_gpu(d.gam2 * sc.dt);
            } else {
                mix_gpu(d.zeta2 * sc.dt);
                flux_gpu();
                substep_gpu(d.gam3 * sc.dt);
            }
            bcon_gpu();
            comm_gpu();
        }

        sc.nit += 1;
        sc.timc += sc.dt;
        sc.timt = sc.timi + sc.timc;
    }

    void step_gpu(MPI_Comm comm, const Params& p, const Derived& d, const Topology& t, GpuSim& g,
                  SimState& scalars) {
        step_gpu(3, comm, p, d, t, g, scalars);
    }

}  // namespace r3d
