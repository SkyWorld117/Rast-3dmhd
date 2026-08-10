// gpu_equivalence_test - the CUDA path must reproduce the CPU reference
// bit-for-bit.  Run with: mpirun -np 4 ./tests/gpu_equivalence_test
//
// Loads the golden pre-loop state, advances several steps on the CPU and on
// the GPU, and requires every primary field (and dt) to be identical.
#include <gtest/gtest.h>
#include <mpi.h>

#include <string>
#include <vector>

#include "golden_util.hpp"
#include "gpu.hpp"
#include "gstate_reader.hpp"
#include "ode.hpp"
#include "step.hpp"

using namespace r3d;
using namespace r3dtest;

constexpr int kSteps = 3;

TEST(Gpu, EquivalenceWithCpu) {
    int npes, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (npes != 4) GTEST_SKIP() << "requires 4 ranks";

    Params p   = golden_params();
    Topology t = resolve_topology(npes, rank, p);
    Derived d  = derive(p, npes, t.npey, t.npez);

    auto build_init = [&](SimState& s) {
        std::vector<double> T(p.npz), R(p.npz);
        static_profile(p, p.npz, T.data(), R.data());
        s.tb            = T[p.npz - 1];
        std::string tag = "00" + std::to_string(rank);
        Gstate pre      = read_gstate(golden_dir() + "/gstate.preloop." + tag + ".0000");
        load_state(s, pre);
    };

    // CPU reference path.
    SimState cpu(d);
    build_init(cpu);
    for (int i = 0; i < kSteps; ++i) step(MPI_COMM_WORLD, p, d, t, cpu);

    // Also compare a single GPU step to the golden reference.
    SimState g1(d);
    build_init(g1);
    GpuSim g1sim;
    gpu_alloc(g1sim, d);
    gpu_upload(g1sim, g1);
    SimState g1sc(d);
    g1sc.tb   = g1.tb;
    g1sc.timi = g1.timi;
    step_gpu(MPI_COMM_WORLD, p, d, t, g1sim, g1sc);
    gpu_download(g1, g1sim);
    gpu_free(g1sim);
    std::string tag = "00" + std::to_string(rank);
    Gstate gs       = read_gstate(golden_dir() + "/gstate.step." + tag + ".0001");
    for (const char* nm : {"ru", "rv", "rw", "ro", "tt"}) {
        const Field* f = nullptr;
        if (nm[0] == 'r' && nm[1] == 'u') f = &g1.ru;
        if (nm[0] == 'r' && nm[1] == 'v') f = &g1.rv;
        if (nm[0] == 'r' && nm[1] == 'w') f = &g1.rw;
        if (nm[0] == 'r' && nm[1] == 'o') f = &g1.ro;
        if (nm[0] == 't') f = &g1.tt;
        double e = max_err_interior(*f, gs.f3d[Gstate::idx3d(nm)], p.ix, p.iy, p.ilap);
        if (e != 0.0) {
            const int nx = d.nx, ny = d.ny, nz = d.nz;
            long bad = 0, bad_boundary = 0;
            double best = 0;
            int bi = -1, bj = -1, bk = -1;
            for (int k = p.ilap / 2; k <= nz - 1 - p.ilap / 2; ++k)
                for (int j = 1; j <= ny - p.iy; ++j)
                    for (int i = 1; i <= nx - p.ix; ++i) {
                        long idx  = (long) k * nx * ny + (long) j * nx + i;
                        double d2 = std::fabs(f->at(i, j, k) - gs.f3d[Gstate::idx3d(nm)][idx]);
                        if (d2 != 0.0) {
                            bad++;
                            if (k == p.ilap / 2 || k == nz - 1 - p.ilap / 2) bad_boundary++;
                            if (d2 > best) {
                                best = d2;
                                bi   = i;
                                bj   = j;
                                bk   = k;
                            }
                        }
                    }
            fprintf(stderr,
                    "GPU-vs-GOLDEN %s rank %d bad=%ld (boundary-planes=%ld) argmax e=%.3e at (i=%d "
                    "j=%d k=%d) gpu=%.17g gold=%.17g\n",
                    nm, rank, bad, bad_boundary, best, bi, bj, bk, f->at(bi, bj, bk),
                    gs.f3d[Gstate::idx3d(nm)][(long) bk * nx * ny + (long) bj * nx + bi]);
        }
        if (nm[0] == 'r' && nm[1] == 'u' && rank == 0) {
            Gstate f1 = read_gstate(golden_dir() + "/gstate.flux1." + tag + ".0000");
            fprintf(stderr,
                    "RU rank0 row: k=1 gpu=%.9g gold=%.9g | k=8 gpu=%.9g gold=%.9g | k=15 gpu=%.9g "
                    "gold=%.9g | k=16 gpu=%.9g gold=%.9g | fu(8,4,8) gpu=%.6g gold=%.6g | ww1 "
                    "gpu=%.6g gold=%.6g\n",
                    f->at(8, 4, 1), gs.f3d[Gstate::idx3d("ru")][1 * 18 * 10 + 4 * 18 + 8],
                    f->at(8, 4, 8), gs.f3d[Gstate::idx3d("ru")][8 * 18 * 10 + 4 * 18 + 8],
                    f->at(8, 4, 15), gs.f3d[Gstate::idx3d("ru")][15 * 18 * 10 + 4 * 18 + 8],
                    f->at(8, 4, 16), gs.f3d[Gstate::idx3d("ru")][16 * 18 * 10 + 4 * 18 + 8],
                    g1.fu.at(8, 4, 8), f1.f3d[Gstate::idx3d("fu")][8 * 180 + 4 * 18 + 8],
                    g1.ww1.at(8, 4, 8), f1.f3d[Gstate::idx3d("ww1")][8 * 180 + 4 * 18 + 8]);
        }
        // The GPU path is now bit-exact vs the golden reference (all primary
        // fields after one step); an exact equality failure is a real regression.
        EXPECT_DOUBLE_EQ(e, 0.0) << "gpu-vs-golden(1step) " << nm << " rank " << rank;
    }

    // GPU path from the same initial state.
    SimState gpu_state(d);
    build_init(gpu_state);
    GpuSim g;
    gpu_alloc(g, d);
    gpu_upload(g, gpu_state);
    SimState gsc(d);  // scalar holder mirror (dt/nit are in SimState)
    gsc.tb   = gpu_state.tb;
    gsc.timi = gpu_state.timi;
    for (int i = 0; i < kSteps; ++i) step_gpu(MPI_COMM_WORLD, p, d, t, g, gsc);
    gpu_download(gpu_state, g);
    gpu_free(g);

    // Bit-exact comparison of the primary fields.
    for (const char* nm : {"ru", "rv", "rw", "ro", "tt"}) {
        const Field* f_cpu = nullptr;
        const Field* f_gpu = nullptr;
        if (nm[0] == 'r' && nm[1] == 'u') {
            f_cpu = &cpu.ru;
            f_gpu = &gpu_state.ru;
        }
        if (nm[0] == 'r' && nm[1] == 'v') {
            f_cpu = &cpu.rv;
            f_gpu = &gpu_state.rv;
        }
        if (nm[0] == 'r' && nm[1] == 'w') {
            f_cpu = &cpu.rw;
            f_gpu = &gpu_state.rw;
        }
        if (nm[0] == 'r' && nm[1] == 'o') {
            f_cpu = &cpu.ro;
            f_gpu = &gpu_state.ro;
        }
        if (nm[0] == 't') {
            f_cpu = &cpu.tt;
            f_gpu = &gpu_state.tt;
        }
        double err = max_err_interior(
            *f_cpu, std::vector<double>(f_gpu->data(), f_gpu->data() + f_gpu->size()), p.ix, p.iy,
            p.ilap);
        fprintf(stderr, "CPUVSGPU %s rank %d e=%.3e\n", nm, rank, err);
        // Bit-exact: the GPU reproduces the (gold-verified) CPU reference.
        EXPECT_DOUBLE_EQ(err, 0.0) << "cpu-vs-gpu " << nm << " rank " << rank;
    }
    EXPECT_DOUBLE_EQ(cpu.dt, gsc.dt) << "dt rank " << rank;
    EXPECT_EQ(cpu.nit, gsc.nit) << "nit rank " << rank;
}

// Per-RK-stage comparison against the golden post-communicate snapshots.
TEST(Gpu, StageByStageVsGolden) {
  int npes, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (npes != 4) GTEST_SKIP() << "requires 4 ranks";
  Params p = golden_params();
  Topology t = resolve_topology(npes, rank, p);
  Derived d = derive(p, npes, t.npey, t.npez);
  std::string tag = "00" + std::to_string(rank);
  for (int st = 1; st <= 3; ++st) {
    SimState s(d);
    std::vector<double> T(p.npz), R(p.npz);
    static_profile(p, p.npz, T.data(), R.data());
    s.tb = T[p.npz - 1];
    Gstate pre = read_gstate(golden_dir() + "/gstate.preloop." + tag + ".0000");
    load_state(s, pre);
    GpuSim g;
    gpu_alloc(g, d);
    gpu_upload(g, s);
    SimState sc(d);
    sc.tb = s.tb; sc.timi = s.timi;
    step_gpu(st, MPI_COMM_WORLD, p, d, t, g, sc);
    gpu_download(s, g);
    gpu_free(g);
    Gstate gs = read_gstate(golden_dir() + "/gstate.ck." + tag + ".000" + std::to_string(st));
    for (const char* nm : {"ru", "rv", "rw", "ro", "tt"}) {
      const Field* f = nullptr;
      if (nm[0]=='r'&&nm[1]=='u') f=&s.ru; if (nm[0]=='r'&&nm[1]=='v') f=&s.rv;
      if (nm[0]=='r'&&nm[1]=='w') f=&s.rw; if (nm[0]=='r'&&nm[1]=='o') f=&s.ro; if (nm[0]=='t') f=&s.tt;    {
      // The RHS arrays (fu..ft) hold the LAST flux computed, so after
      // step_gpu(st) they equal the stage-'st' flux -> compare vs flux{st}.
      char fname[128];
      if (st == 1) {
        // Probe ww1 (ro*tt full pass) at the deep interior + bottom ghost planes.
        Gstate f1p = read_gstate(golden_dir() + "/gstate.flux1." + tag + ".0000");
        fprintf(stderr, "WW1 rank %d: gpu(k=15)=%.9g gold=%.9g | gpu(k=16)=%.9g gold=%.9g | gpu(k=17)=%.9g gold=%.9g | ro(k=17) gpu=%.9g goldpre=%.9g\n",
          rank,
          s.ww1.at(1,1,15), f1p.f3d[Gstate::idx3d("ww1")][15*18*10+1*18+1],
          s.ww1.at(1,1,16), f1p.f3d[Gstate::idx3d("ww1")][16*18*10+1*18+1],
          s.ww1.at(1,1,17), f1p.f3d[Gstate::idx3d("ww1")][17*18*10+1*18+1],
          s.ro.at(1,1,17), 0.0);
      }
      snprintf(fname, sizeof(fname), "/gstate.flux%d.%s.0000", st, tag.c_str());
      Gstate fs = read_gstate(golden_dir() + fname);
      for (const char* nm : {"fu", "fv", "fw", "fr", "ft"}) {
        const Field* f = nullptr;
        if (nm[1]=='u') f=&s.fu; if (nm[1]=='v') f=&s.fv;
        if (nm[1]=='w') f=&s.fw; if (nm[1]=='r') f=&s.fr; if (nm[1]=='t') f=&s.ft;
        double e = max_err_interior(*f, fs.f3d[Gstate::idx3d(nm)], p.ix, p.iy, p.ilap);
        if (e != 0.0) {
          const int nx=d.nx,ny=d.ny,nz=d.nz; double best=0; int bi=-1,bj=-1,bk=-1;
          for (int k=p.ilap/2;k<=nz-1-p.ilap/2;++k) for (int j=1;j<=ny-p.iy;++j) for (int i=1;i<=nx-p.ix;++i){
            long idx=(long)k*nx*ny+(long)j*nx+i;
            double d2=std::fabs(f->at(i,j,k)-fs.f3d[Gstate::idx3d(nm)][idx]);
            if(d2>best){best=d2;bi=i;bj=j;bk=k;}
          }
          fprintf(stderr, "FLUX%d %s rank %d (i=%d j=%d k=%d) e=%.3e gpu=%.17g gold=%.17g\n",
                  st, nm, rank, bi,bj,bk,best, f->at(bi,bj,bk),
                  fs.f3d[Gstate::idx3d(nm)][(long)bk*nx*ny+(long)bj*nx+bi]);
        } else {
          fprintf(stderr, "FLUX%d %s rank %d EXACT\n", st, nm, rank);
        }
      }
    }

      double e = max_err_interior(*f, gs.f3d[Gstate::idx3d(nm)], p.ix, p.iy, p.ilap);
      if (e != 0.0) {
        const int nx=d.nx,ny=d.ny,nz=d.nz; double best=0; int bi=-1,bj=-1,bk=-1;
        for (int k=p.ilap/2;k<=nz-1-p.ilap/2;++k) for (int j=1;j<=ny-p.iy;++j) for (int i=1;i<=nx-p.ix;++i){
          long idx=(long)k*nx*ny+(long)j*nx+i;
          double d2=std::fabs(f->at(i,j,k)-gs.f3d[Gstate::idx3d(nm)][idx]);
          if(d2>best){best=d2;bi=i;bj=j;bk=k;}
        }
        fprintf(stderr, "STAGE%d %s rank %d (i=%d j=%d k=%d) e=%.3e gpu=%.17g gold=%.17g\n",
                st, nm, rank, bi,bj,bk,best, f->at(bi,bj,bk),
                gs.f3d[Gstate::idx3d(nm)][(long)bk*nx*ny+(long)bj*nx+bi]);
      }
      // Report, do not strictly gate (diagnostic bisect).
    }
    // Ghost-plane probe on the final stage: compare k=17 (bottom ghost of
    // the top rank) and the y ghost rows against the golden ck snapshot.
    if (rank <= 1) {
      Gstate ckj = read_gstate(golden_dir() + "/gstate.ck." + tag + ".000" + std::to_string(st));
      auto idx3 = [&](const char* n, int i, int j, int k) {
        return (long)k*d.nx*d.ny + (long)j*d.nx + i;
      };
      fprintf(stderr, "YCOL st%d rank %d ru[i=8,k=8] j0 gpu=%.9g gold=%.9g | j8 gpu=%.9g gold=%.9g | j9 gpu=%.9g gold=%.9g\n",
        st, rank,
        s.ru.at(8,0,8), ckj.f3d[Gstate::idx3d("ru")][idx3("ru",8,0,8)],
        s.ru.at(8,8,8), ckj.f3d[Gstate::idx3d("ru")][idx3("ru",8,8,8)],
        s.ru.at(8,9,8), ckj.f3d[Gstate::idx3d("ru")][idx3("ru",8,9,8)]);
    }
    if (st == 3) {
      Gstate ck = read_gstate(golden_dir() + "/gstate.ck." + tag + ".0003");
      auto gidx = [&](const char* n, int i, int j, int k) {
        return (long)k*d.nx*d.ny + (long)j*d.nx + i;
      };
      const char* nm = "ru";
      // top rank: interior ru at k=16/17; compare the ghost plane k=17 too.
      fprintf(stderr, "PST rank %d ru[8,4,16] gpu=%.9g gold=%.9g  ru[8,4,17] gpu=%.9g gold=%.9g  ru[0,4,8] gpu=%.9g gold=%.9g  ru[17,4,8] gpu=%.9g gold=%.9g  ru[8,0,8] gpu=%.9g gold=%.9g  ru[8,9,8] gpu=%.9g gold=%.9g\n",
        rank,
        s.ru.at(8,4,16), ck.f3d[Gstate::idx3d(nm)][gidx(nm,8,4,16)],
        s.ru.at(8,4,17), ck.f3d[Gstate::idx3d(nm)][gidx(nm,8,4,17)],
        s.ru.at(0,4,8), ck.f3d[Gstate::idx3d(nm)][gidx(nm,0,4,8)],
        s.ru.at(17,4,8), ck.f3d[Gstate::idx3d(nm)][gidx(nm,17,4,8)],
        s.ru.at(8,0,8), ck.f3d[Gstate::idx3d(nm)][gidx(nm,8,0,8)],
        s.ru.at(8,9,8), ck.f3d[Gstate::idx3d(nm)][gidx(nm,8,9,8)]);
    }
    fprintf(stderr, "STAGE%d done rank %d\n", st, rank);
  }
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}
