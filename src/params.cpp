// params.cpp - Derived computation and CLI parsing for the runtime config.
#include "params.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "numcompat.hpp"

namespace r3d {

    Derived derive(const Params& p, int np, int npey, int npez) {
        Derived d;
        d.np   = np;
        d.npey = npey;
        d.npez = npez;
        d.nry  = p.npy / npey;
        d.nrz  = p.npz / npez;
        d.nx   = p.npx + p.ix;
        d.ny   = d.nry + p.iy;
        d.nz   = d.nrz + p.ilap;

        // The original computes these from single-precision literals and FLOAT()
        // (== float), then widens:  1.0E00/3.0E00 first becomes 0.33333334f.
        // f4() reproduces that.  Products with already-double values (C23=2.0E00*C13)
        // stay double after the first seed, so they use the widened seed directly.
        d.c13 = f4(1.0f / 3.0f);
        d.c23 = 2.0 * d.c13;
        d.c43 = 4.0 * d.c13;

        d.gam1  = f4(8.0f / 15.0f);
        d.gam2  = f4(5.0f / 12.0f);
        d.gam3  = f4(3.0f / 4.0f);
        d.zeta1 = f4(-17.0f / 60.0f);
        d.zeta2 = f4(-5.0f / 12.0f);

        d.hx  = f4(0.5f * (float) (p.npx - 1));
        d.h2x = f4((float) (p.npx - 1) * (float) (p.npx - 1));
        d.hy  = f4(0.5f * (float) (p.npy - 1));
        d.h2y = f4((float) (p.npy - 1) * (float) (p.npy - 1));
        d.hz  = f4(0.5f * (float) (p.npz - 1));
        d.h2z = f4((float) (p.npz - 1) * (float) (p.npz - 1));

        d.repr   = p.re * p.pr;
        d.ore    = 1.0 / p.re;
        d.cv     = 1.0 / (p.gamma - 1.0);
        d.ocv    = 1.0 / d.cv;
        d.rkapst = (p.grav == 0.0) ? 1.0 : (p.polys + 1.0) * p.theta / p.grav;
        if (p.lmag) {
            d.orm   = 1.0 / p.rm;
            d.obeta = 2.0 / p.beta;
        } else {
            d.orm   = 0.0;
            d.obeta = 0.0;
        }
        if (p.lrot) {
            d.omx = sin(p.ang) / p.r_y;
            d.omz = cos(p.ang) / p.r_y;
        } else {
            d.omx = 0.0;
            d.omz = 0.0;
        }
        return d;
    }

    // ---------------------------------------------------------------- CLI ----
    namespace {

        [[noreturn]] void usage_error(const std::string& msg) {
            fprintf(stderr, "rast-3dmhd: %s\n", msg.c_str());
            print_help();
            exit(EXIT_FAILURE);
        }

        bool parse_bool(const std::string& s) {
            return s == "1" || s == "true" || s == "yes" || s == "on";
        }

    }  // namespace

    int parse_args(int argc, char** argv, Params& p) {
        // Accept "--key value" and "--key=value".
        auto take = [&](int& i, const char* key) -> std::string {
            std::string arg = argv[i];
            auto eq         = arg.find('=');
            if (eq != std::string::npos) return arg.substr(eq + 1);
            if (i + 1 >= argc) usage_error(std::string("missing value for ") + key);
            return argv[++i];
        };

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help" || arg == "-h" || arg == "-?") {
                print_help();
                exit(EXIT_SUCCESS);
            }
            auto flag = [&](const char* name) {
                return arg == name || (arg.rfind(name, 0) == 0 && arg.size() > strlen(name) &&
                                       arg[strlen(name)] == '=');
            };
#define INT_FLAG(name, var)                       \
    if (flag(name)) {                             \
        p.var = std::atoi(take(i, name).c_str()); \
        continue;                                 \
    }
#define DBL_FLAG(name, var)                       \
    if (flag(name)) {                             \
        p.var = std::atof(take(i, name).c_str()); \
        continue;                                 \
    }
            INT_FLAG("--npx", npx)
            INT_FLAG("--npy", npy)
            INT_FLAG("--npz", npz)
            INT_FLAG("--npey", npey)
            INT_FLAG("--ix", ix)
            INT_FLAG("--iy", iy)
            INT_FLAG("--ilap", ilap)
            INT_FLAG("--ngrid", ngrid)
            INT_FLAG("--id", id)
            INT_FLAG("--ncol", ncol)
            INT_FLAG("--nrow", nrow)
            INT_FLAG("--ixcon", ixcon)
            INT_FLAG("--iycon", iycon)
            INT_FLAG("--izcon", izcon)
            INT_FLAG("--itcon", itcon)
            INT_FLAG("--ibcon", ibcon)
            INT_FLAG("--ncase", ncase)
            INT_FLAG("--ncasep", ncasep)
            INT_FLAG("--ntotal", ntotal)
            INT_FLAG("--nstep0", nstep0)
            INT_FLAG("--nstart", nstart)
            INT_FLAG("--ntube", ntube)
            INT_FLAG("--gpu", gpu_id)
            DBL_FLAG("--re", re)
            DBL_FLAG("--pr", pr)
            DBL_FLAG("--theta", theta)
            DBL_FLAG("--grav", grav)
            DBL_FLAG("--ry", r_y)
            DBL_FLAG("--ang", ang)
            DBL_FLAG("--rm", rm)
            DBL_FLAG("--beta", beta)
            DBL_FLAG("--pzp", pzp)
            DBL_FLAG("--sigma", sigma)
            DBL_FLAG("--polys", polys)
            DBL_FLAG("--tp", tp)
            DBL_FLAG("--tc", tc)
            DBL_FLAG("--xp", xp)
            DBL_FLAG("--yp", yp)
            DBL_FLAG("--zp", zp)
            DBL_FLAG("--xmax", xmax)
            DBL_FLAG("--ymax", ymax)
            DBL_FLAG("--zmax", zmax)
            DBL_FLAG("--ampt", ampt)
            DBL_FLAG("--ampb", ampb)
            DBL_FLAG("--bfh", bfh)
            DBL_FLAG("--bzp", bzp)
            DBL_FLAG("--qfh", qfh)
            DBL_FLAG("--gamma", gamma)
            DBL_FLAG("--hh", hh)
            DBL_FLAG("--dh", dh)
            DBL_FLAG("--dv", dv)
            DBL_FLAG("--tstart", tstart)
            DBL_FLAG("--toff", toff)
            DBL_FLAG("--rlax", rlax)
            DBL_FLAG("--xx1", xx1)
            DBL_FLAG("--xx2", xx2)
            DBL_FLAG("--yy1", yy1)
            DBL_FLAG("--yy2", yy2)
            DBL_FLAG("--zz1", zz1)
            DBL_FLAG("--zz2", zz2)
            DBL_FLAG("--xa", xa)
            DBL_FLAG("--xb", xb)
            DBL_FLAG("--xc", xc)
            DBL_FLAG("--xd", xd)
            DBL_FLAG("--ya", ya)
            DBL_FLAG("--yb", yb)
            DBL_FLAG("--yc", yc)
            DBL_FLAG("--yd", yd)
            DBL_FLAG("--za", za)
            DBL_FLAG("--zb", zb)
            DBL_FLAG("--zc", zc)
            DBL_FLAG("--zd", zd)
            DBL_FLAG("--xcent", xcent)
            DBL_FLAG("--zcent", zcent)
            DBL_FLAG("--rmax", rmax)
            DBL_FLAG("--cmt", cmt)
            DBL_FLAG("--a", a_tube)
            DBL_FLAG("--lambda", lambda)
            DBL_FLAG("--vz0", vz0)
            DBL_FLAG("--sff", sf)
#undef INT_FLAG
#undef DBL_FLAG
            if (flag("--lrot")) {
                p.lrot = parse_bool(take(i, "--lrot"));
                continue;
            }
            if (flag("--lmag")) {
                p.lmag = parse_bool(take(i, "--lmag"));
                continue;
            }
            if (flag("--lpot")) {
                p.lpot = parse_bool(take(i, "--lpot"));
                continue;
            }
            if (flag("--lrem")) {
                p.lrem = parse_bool(take(i, "--lrem"));
                continue;
            }
            if (flag("--lshr")) {
                p.lshr = parse_bool(take(i, "--lshr"));
                continue;
            }
            if (flag("--backend")) {
                std::string b = take(i, "--backend");
                if (b == "cpu")
                    p.backend = Params::Backend::kCpu;
                else if (b == "gpu")
                    p.backend = Params::Backend::kGpu;
                else
                    usage_error("--backend must be 'cpu' or 'gpu'");
                continue;
            }
            usage_error("unknown option '" + arg + "'");
        }
        return 0;
    }

    void print_help() {
        // clang-format off
  fprintf(stderr,
    "rast-3dmhd - GPU-accelerated 3D magnetohydrodynamics (C++/CUDA rewrite of the\n"
    "Rast 3dmhd Fortran code).  MPI must be initialised before this parser runs;\n"
    "run under mpirun with at least 2 ranks (the code requires NPEZ >= 2).\n"
    "\n"
    "Usage: mpirun -np <NPE> 3dmhd [options]\n"
    "\n"
    "Geometry / topology\n"
    "  --npx N   --npy N   --npz N    global grid points (default 504 504 2048)\n"
    "  --npey N                          processors in y (default 8; npez=NPE/npey)\n"
    "  --ngrid {0,1,2}                 grid stretch function (default 1)\n"
    "  --ix 2 --iy 2 --ilap 2          stencil widths (ghosts)\n"
    "  --ixcon 0 --iycon 0 --izcon 0 --itcon 1 --ibcon 0   boundary conditions\n"
    "\n"
    "Physics\n"
    "  --re R --pr R --theta R --grav R    (default 2 0.05 0.25 0.625)\n"
    "  --rm R --beta R --ampb R --bfh R --bzp R   (magnetic layer)\n"
    "  --ampt R    initial temperature perturbation amplitude\n"
    "  --gamma R   Cp/Cv (default 5/3)\n"
    "  --xmax R --ymax R --zmax R  (default 20 20 40)\n"
    "  --sff R     time-step safety factor (default 0.315)\n"
    "  --lrot 0|1 --lmag 0|1 --lpot 0|1 --lrem 0|1 --lshr 0|1\n"
    "  --ntube N   tube model (0..5)      --ntotal N --nstep0 N --nstart N\n"
    "\n"
    "Grid stretch coefficients\n"
    "  --xx1 R --xx2 R --yy1 R --yy2 R --zz1 R --zz2 R\n"
    "  --xa R --xb R --xc R --xd R (and --ya/--yb/--yc/--yd, --za/--zb/--zc/--zd)\n"
    "\n"
    "Execution\n"
    "  --backend cpu|gpu   (default cpu; 'gpu' requires a CUDA build)\n"
    "  --gpu ID            CUDA device to use per rank (default 0)\n"
    "  --help              this message\n");
        // clang-format on
    }

}  // namespace r3d
