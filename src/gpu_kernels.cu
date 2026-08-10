// gpu_kernels.cu - CUDA kernels mirroring the CPU FLUXES / STEP / BCON
// sweeps (fluxes.cpp / step.cpp / bcon.cpp), sweep-for-sweep, so the GPU
// result is bit-identical to the golden-validated CPU path.  Every kernel is
// launched over exactly the same cell range as its CPU loop.
#include <cuda_runtime.h>
#include <math.h>

namespace r3d {

    struct K {
        int nx, ny, nz;
        int i1, i2, j1, j2, k1, k2;  // interior (inclusive)
        double c13, c23, c43, ore, ocv, repr, grav;
        double hx, h2x, hy, h2y, hz, h2z;
        double omx, omz;
        double gamma, re, cv;
        double xp, yp, hh;
        int li, lj, lk;  // ilap/2, iy/2, ix/2
    };

    __device__ __forceinline__ long f3(int nx, int ny, int i, int j, int k) {
        return (long) k * nx * ny + (long) j * nx + i;
    }

    // ---------------------------------------------------------------------------
    // FLUXES
    // ---------------------------------------------------------------------------
    // FR from the momentum divergence (interior); WW1 = dRW/dz there.
    __global__ void k_fr(const K q, const double* ru, const double* rv, const double* rw,
                         double* fr, double* ww1, const double* dxxdx, const double* dyydy,
                         const double* dzzdz) {
        long n = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i  = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r = t / (q.i2 - q.i1 + 1);
            int j  = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k  = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long f = f3(q.nx, q.ny, i, j, k);
            fr[f]  = (ru[f3(q.nx, q.ny, i - 1, j, k)] - ru[f3(q.nx, q.ny, i + 1, j, k)]) * q.hx *
                     dxxdx[i];
            fr[f] -= (rv[f3(q.nx, q.ny, i, j + 1, k)] - rv[f3(q.nx, q.ny, i, j - 1, k)]) * q.hy *
                     dyydy[j];
            ww1[f] = (rw[f3(q.nx, q.ny, i, j, k + 1)] - rw[f3(q.nx, q.ny, i, j, k - 1)]) * q.hz *
                     dzzdz[k];
        }
    }
    __global__ void k_fr_w1b(const K q, const double* rw, double* ww1, const double* dzzdz,
                             int top) {
        long n    = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1);
        int k     = top ? q.k1 : q.k2;
        double tz = q.hz * dzzdz[k];
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i  = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            int j  = q.j1 + (int) (t / (q.i2 - q.i1 + 1));
            long f = f3(q.nx, q.ny, i, j, k);
            ww1[f] =
                top ? (4.0 * rw[f3(q.nx, q.ny, i, j, k + 1)] - rw[f3(q.nx, q.ny, i, j, k + 2)]) * tz
                    : (rw[f3(q.nx, q.ny, i, j, k - 2)] - 4.0 * rw[f3(q.nx, q.ny, i, j, k - 1)]) *
                          tz;
        }
    }
    __global__ void k_fr_subw1(const K q, double* fr, const double* ww1) {
        long n = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i  = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r = t / (q.i2 - q.i1 + 1);
            int j  = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k  = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long f = f3(q.nx, q.ny, i, j, k);
            fr[f] -= ww1[f];
        }
    }
    // FW=GRAV*RO (interior); full-array WW1=RO*TT and RO=1/RO; pressure grads.
    __global__ void k_force_prep(const K q, const double* ro0, const double* tt, double* ro,
                                 double* ww1) {
        long full = (long) q.nx * q.ny * q.nz;
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < full;
             t += (long) blockDim.x * gridDim.x) {
            double v = ro0[t];
            ww1[t]   = v * tt[t];
            ro[t]    = 1.0 / v;
        }
    }
    __global__ void k_force(const K q, double grav, const double* ro0, double* fu, double* fv,
                            double* fw, const double* ww1, const double* dxxdx, const double* dyydy,
                            const double* dzzdz) {
        long n = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i     = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r    = t / (q.i2 - q.i1 + 1);
            int j     = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k     = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long f    = f3(q.nx, q.ny, i, j, k);
            double ty = q.hy * dyydy[j];
            double tz = q.hz * dzzdz[k];
            fw[f]     = grav * ro0[f];
            fw[f] -= (ww1[f3(q.nx, q.ny, i, j, k + 1)] - ww1[f3(q.nx, q.ny, i, j, k - 1)]) * tz;
            fv[f] = (ww1[f3(q.nx, q.ny, i, j - 1, k)] - ww1[f3(q.nx, q.ny, i, j + 1, k)]) * ty;
            fu[f] = (ww1[f3(q.nx, q.ny, i - 1, j, k)] - ww1[f3(q.nx, q.ny, i + 1, j, k)]) * q.hx *
                    dxxdx[i];
        }
    }
    __global__ void k_vel(const K q, const double* ru, const double* rv, const double* rw,
                          const double* ro, double* uu, double* vv, double* ww) {
        long full = (long) q.nx * q.ny * q.nz;
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < full;
             t += (long) blockDim.x * gridDim.x) {
            uu[t] = ru[t] * ro[t];
            vv[t] = rv[t] * ro[t];
            ww[t] = rw[t] * ro[t];
        }
    }
    // Advection of the momenta (single kernel, selects field by im).
    __global__ void k_adv(const K q, int im, const double* ru, const double* rv, const double* rw,
                          const double* uu, const double* vv, const double* ww, double* fu,
                          double* fv, double* fw, const double* dxxdx, const double* dyydy,
                          const double* dzzdz) {
        const double* m = im == 0 ? ru : (im == 1 ? rv : rw);
        double* f       = im == 0 ? fu : (im == 1 ? fv : fw);
        long n          = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i     = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r    = t / (q.i2 - q.i1 + 1);
            int j     = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k     = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long c    = f3(q.nx, q.ny, i, j, k);
            double ty = q.hy * dyydy[j];
            double tz = q.hz * dzzdz[k];
            f[c] -= (m[f3(q.nx, q.ny, i + 1, j, k)] * uu[f3(q.nx, q.ny, i + 1, j, k)] -
                     m[f3(q.nx, q.ny, i - 1, j, k)] * uu[f3(q.nx, q.ny, i - 1, j, k)]) *
                    q.hx * dxxdx[i];
            f[c] -= (m[f3(q.nx, q.ny, i, j + 1, k)] * vv[f3(q.nx, q.ny, i, j + 1, k)] -
                     m[f3(q.nx, q.ny, i, j - 1, k)] * vv[f3(q.nx, q.ny, i, j - 1, k)]) *
                    ty;
            f[c] -= (m[f3(q.nx, q.ny, i, j, k + 1)] * ww[f3(q.nx, q.ny, i, j, k + 1)] -
                     m[f3(q.nx, q.ny, i, j, k - 1)] * ww[f3(q.nx, q.ny, i, j, k - 1)]) *
                    tz;
        }
    }
    __global__ void k_ft_advect(const K q, const double* uu, const double* vv, const double* ww,
                                const double* tt, double* ft, const double* dxxdx,
                                const double* dyydy, const double* dzzdz) {
        long n = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i     = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r    = t / (q.i2 - q.i1 + 1);
            int j     = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k     = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long c    = f3(q.nx, q.ny, i, j, k);
            double ty = q.hy * dyydy[j];
            double tz = q.hz * dzzdz[k];
            ft[c]     = -1.0 * uu[c] *
                        (tt[f3(q.nx, q.ny, i + 1, j, k)] - tt[f3(q.nx, q.ny, i - 1, j, k)]) * q.hx *
                        dxxdx[i];
            ft[c] -=
                vv[c] * (tt[f3(q.nx, q.ny, i, j + 1, k)] - tt[f3(q.nx, q.ny, i, j - 1, k)]) * ty;
            ft[c] -=
                ww[c] * (tt[f3(q.nx, q.ny, i, j, k + 1)] - tt[f3(q.nx, q.ny, i, j, k - 1)]) * tz;
        }
    }
    __global__ void k_ft_diffuse(const K q, const double* ro, const double* tt, double* ft,
                                 const double* dxxdx, const double* d2xxdx2, const double* dyydy,
                                 const double* d2yydy2, const double* dzzdz, const double* d2zzdz2,
                                 const double* rkapa, const double* dkapa) {
        const double tmp = q.ocv / q.repr;
        long n           = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i      = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r     = t / (q.i2 - q.i1 + 1);
            int j      = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k      = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long c     = f3(q.nx, q.ny, i, j, k);
            double tz1 = q.hz * d2zzdz2[k];
            double tz2 = q.h2z * dzzdz[k] * dzzdz[k];
            double ty1 = q.hy * d2yydy2[j];
            double ty2 = q.h2y * dyydy[j] * dyydy[j];
            double xm = tt[f3(q.nx, q.ny, i - 1, j, k)], xp = tt[f3(q.nx, q.ny, i + 1, j, k)];
            double ym = tt[f3(q.nx, q.ny, i, j - 1, k)], yp = tt[f3(q.nx, q.ny, i, j + 1, k)];
            double zm = tt[f3(q.nx, q.ny, i, j, k - 1)], zp = tt[f3(q.nx, q.ny, i, j, k + 1)];
            ft[c] += ((xp - xm) * q.hx * d2xxdx2[i] +
                      (xp - 2.0 * tt[c] + xm) * q.h2x * dxxdx[i] * dxxdx[i]) *
                     ro[c] * tmp * rkapa[k];
            ft[c] += ((yp - ym) * ty1 + (yp - 2.0 * tt[c] + ym) * ty2) * ro[c] * tmp * rkapa[k];
            ft[c] += ((zp - zm) * tz1 + (zp - 2.0 * tt[c] + zm) * tz2) * ro[c] * tmp * rkapa[k];
            ft[c] += (zp - zm) * q.hz * dzzdz[k] * ro[c] * tmp * dkapa[k];
        }
    }
    // Viscous momentum diffusion (C: 0=fU,1=fV,2=fW).
    template <int C>
    __device__ void k_viscous(const K q, const double* uu, const double* vv, const double* ww,
                              double* fu, double* fv, double* fw, const double* dxxdx,
                              const double* d2xxdx2, const double* dyydy, const double* d2yydy2,
                              const double* dzzdz, const double* d2zzdz2) {
        const double* v = C == 0 ? uu : (C == 1 ? vv : ww);
        double* f       = C == 0 ? fu : (C == 1 ? fv : fw);
        long n          = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i      = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r     = t / (q.i2 - q.i1 + 1);
            int j      = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k      = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long c     = f3(q.nx, q.ny, i, j, k);
            double tz1 = q.hz * d2zzdz2[k];
            double tz2 = q.h2z * dzzdz[k] * dzzdz[k];
            double ty1 = q.hy * d2yydy2[j];
            double ty2 = q.h2y * dyydy[j] * dyydy[j];
            double xm = v[f3(q.nx, q.ny, i - 1, j, k)], xp = v[f3(q.nx, q.ny, i + 1, j, k)];
            double ym = v[f3(q.nx, q.ny, i, j - 1, k)], yp = v[f3(q.nx, q.ny, i, j + 1, k)];
            double zm = v[f3(q.nx, q.ny, i, j, k - 1)], zp = v[f3(q.nx, q.ny, i, j, k + 1)];
            double X  = (xp - xm) * q.hx * d2xxdx2[i] +
                        (xp - 2.0 * v[c] + xm) * q.h2x * dxxdx[i] * dxxdx[i];
            double Y1 = (yp - ym) * ty1;
            double Y2 = (yp - 2.0 * v[c] + ym) * ty2;
            double Z1 = (zp - zm) * tz1;
            double Z2 = (zp - 2.0 * v[c] + zm) * tz2;
            // Left-to-right association identical to the (bit-exact) CPU port.
            if (C == 0)
                f[c] = f[c] + q.ore * (q.c43 * X + Y1 + Y2 + Z1 + Z2);
            else if (C == 1)
                f[c] = f[c] + q.ore * (X + q.c43 * (Y1 + Y2) + Z1 + Z2);
            else
                f[c] = f[c] + q.ore * (X + Y1 + Y2 + q.c43 * (Z1 + Z2));
        }
    }
    // WW1=dU/dx (J=1..NY full), WW2=dV/dx (J=1..NY full).
    __global__ void k_w1_w2_dx(const K q, const double* uu, const double* vv, double* ww1,
                               double* ww2, const double* dxxdx) {
        long n = (long) (q.i2 - q.i1 + 1) * q.ny * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i  = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r = t / (q.i2 - q.i1 + 1);
            int j  = (int) (r % q.ny);
            int k  = q.k1 + (int) (r / q.ny);
            long c = f3(q.nx, q.ny, i, j, k);
            ww1[c] = (uu[f3(q.nx, q.ny, i + 1, j, k)] - uu[f3(q.nx, q.ny, i - 1, j, k)]) * q.hx *
                     dxxdx[i];
            ww2[c] = (vv[f3(q.nx, q.ny, i + 1, j, k)] - vv[f3(q.nx, q.ny, i - 1, j, k)]) * q.hx *
                     dxxdx[i];
        }
    }
    __global__ void k_w3_dx(const K q, const double* ww, double* ww3, const double* dxxdx) {
        long n = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i  = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r = t / (q.i2 - q.i1 + 1);
            int j  = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k  = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long c = f3(q.nx, q.ny, i, j, k);
            ww3[c] = (ww[f3(q.nx, q.ny, i + 1, j, k)] - ww[f3(q.nx, q.ny, i - 1, j, k)]) * q.hx *
                     dxxdx[i];
        }
    }
    // FT += (C43*WW1^2+WW2^2+WW3^2)*RO*TMP ; FT -= OCV*WW1*TT
    __global__ void k_ft_heat1(const K q, const double* ro, const double* tt, double* ft,
                               const double* ww1, const double* ww2, const double* ww3) {
        const double tmp = q.ocv * q.ore;
        long n           = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i  = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r = t / (q.i2 - q.i1 + 1);
            int j  = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k  = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long c = f3(q.nx, q.ny, i, j, k);
            ft[c] += (q.c43 * ww1[c] * ww1[c] + ww2[c] * ww2[c] + ww3[c] * ww3[c]) * ro[c] * tmp;
            ft[c] -= q.ocv * ww1[c] * tt[c];
        }
    }
    __global__ void k_cross_fufv(const K q, double* fu, double* fv, const double* ww1,
                                 const double* ww2, const double* dyydy) {
        const double tmpc = q.c13 * q.ore;
        long n            = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i  = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r = t / (q.i2 - q.i1 + 1);
            int j  = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k  = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long c = f3(q.nx, q.ny, i, j, k);
            fu[c] += tmpc * (ww2[f3(q.nx, q.ny, i, j + 1, k)] - ww2[f3(q.nx, q.ny, i, j - 1, k)]) *
                     q.hy * dyydy[j];
            fv[c] += tmpc * (ww1[f3(q.nx, q.ny, i, j + 1, k)] - ww1[f3(q.nx, q.ny, i, j - 1, k)]) *
                     q.hy * dyydy[j];
        }
    }
    // WW1=dU/dy, WW2=dV/dy (interior j)
    // WW1 = dU/dy (interior).  Split from the WW2 update because the heat-2
    // term (WW1^2 + 2*WW1*WW2) must use WW2 = dV/dx (still valid) - the
    // Fortran only recomputes WW2 = dV/dy AFTER that first accumulation.
    __global__ void k_w1_dy(const K q, const double* uu, double* ww1, const double* dyydy) {
        long n = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i  = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r = t / (q.i2 - q.i1 + 1);
            int j  = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k  = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long c = f3(q.nx, q.ny, i, j, k);
            ww1[c] = (uu[f3(q.nx, q.ny, i, j + 1, k)] - uu[f3(q.nx, q.ny, i, j - 1, k)]) * q.hy *
                     dyydy[j];
        }
    }
    // WW2 = dV/dy (interior), computed only after the 2*WW1*WW2 accumulation.
    __global__ void k_w2_dy(const K q, const double* vv, double* ww2, const double* dyydy) {
        long n = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i  = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r = t / (q.i2 - q.i1 + 1);
            int j  = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k  = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long c = f3(q.nx, q.ny, i, j, k);
            ww2[c] = (vv[f3(q.nx, q.ny, i, j + 1, k)] - vv[f3(q.nx, q.ny, i, j - 1, k)]) * q.hy *
                     dyydy[j];
        }
    }
    // FT += (WW1^2 + 2*WW1*WW2)*RO*OCV*ORE  (WW1=dU/dy, WW2 = dV/dx).
    __global__ void k_ft_heat2a(const K q, const double* ro, double* ft, const double* ww1,
                                const double* ww2) {
        const double tmp = q.ocv * q.ore;
        long n           = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i  = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r = t / (q.i2 - q.i1 + 1);
            int j  = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k  = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long c = f3(q.nx, q.ny, i, j, k);
            ft[c] += (ww1[c] * ww1[c] + 2.0 * ww1[c] * ww2[c]) * ro[c] * tmp;
        }
    }
    // FT += C43*WW2^2*RO*OCV*ORE ; FT -= OCV*WW2*TT  (WW2=dV/dy).
    __global__ void k_ft_heat2b(const K q, const double* ro, const double* tt, double* ft,
                                const double* ww2) {
        const double tmp43 = q.c43 * q.ocv * q.ore;
        long n             = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i  = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r = t / (q.i2 - q.i1 + 1);
            int j  = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k  = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long c = f3(q.nx, q.ny, i, j, k);
            ft[c] += ww2[c] * ww2[c] * ro[c] * tmp43;
            ft[c] -= q.ocv * ww2[c] * tt[c];
        }
    }
    // Recompute WW1 = dU/dx (interior), then WW3 = dW/dz (full i,j).
    __global__ void k_w1_dx_again(const K q, const double* uu, double* ww1, const double* dxxdx) {
        long n = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i  = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r = t / (q.i2 - q.i1 + 1);
            int j  = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k  = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long c = f3(q.nx, q.ny, i, j, k);
            ww1[c] = (uu[f3(q.nx, q.ny, i + 1, j, k)] - uu[f3(q.nx, q.ny, i - 1, j, k)]) * q.hx *
                     dxxdx[i];
        }
    }
    __global__ void k_w3_dz_full(const K q, const double* ww, double* ww3, const double* dzzdz) {
        long n = (long) q.nx * q.ny * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i  = (int) (t % q.nx);
            long r = t / q.nx;
            int j  = (int) (r % q.ny);
            int k  = q.k1 + (int) (r / q.ny);
            long c = f3(q.nx, q.ny, i, j, k);
            ww3[c] = (ww[f3(q.nx, q.ny, i, j, k + 1)] - ww[f3(q.nx, q.ny, i, j, k - 1)]) * q.hz *
                     dzzdz[k];
        }
    }
    __global__ void k_ft_heat3a(const K q, const double* ro, const double* tt, double* ft,
                                const double* ww1, const double* ww2, const double* ww3) {
        const double tmp = q.c43 * q.ocv * q.ore;
        long n           = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i  = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r = t / (q.i2 - q.i1 + 1);
            int j  = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k  = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long c = f3(q.nx, q.ny, i, j, k);
            ft[c] += (ww3[c] * ww3[c] - ww1[c] * ww3[c] - ww1[c] * ww2[c] - ww2[c] * ww3[c]) *
                     ro[c] * tmp;
            ft[c] -= q.ocv * ww3[c] * tt[c];
        }
    }
    // FU += C13*ORE*(dWW3/dx); FV += C13*ORE*(dWW3/dy)
    __global__ void k_cross_w3(const K q, double* fu, double* fv, const double* ww3,
                               const double* dxxdx, const double* dyydy) {
        const double tmpc = q.c13 * q.ore;
        long n            = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i  = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r = t / (q.i2 - q.i1 + 1);
            int j  = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k  = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long c = f3(q.nx, q.ny, i, j, k);
            fu[c] += tmpc * (ww3[f3(q.nx, q.ny, i + 1, j, k)] - ww3[f3(q.nx, q.ny, i - 1, j, k)]) *
                     q.hx * dxxdx[i];
            fv[c] += tmpc * (ww3[f3(q.nx, q.ny, i, j + 1, k)] - ww3[f3(q.nx, q.ny, i, j - 1, k)]) *
                     q.hy * dyydy[j];
        }
    }
    // WW1=dU/dz (I=1..NX), WW2=dV/dz (J=1..NY)
    __global__ void k_w1_w2_dz(const K q, const double* uu, const double* vv, double* ww1,
                               double* ww2, const double* dzzdz) {
        long n1 = (long) q.nx * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        long n2 = (long) (q.i2 - q.i1 + 1) * q.ny * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n1 + n2;
             t += (long) blockDim.x * gridDim.x) {
            if (t < n1) {
                int i  = (int) (t % q.nx);
                long r = t / q.nx;
                int j  = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
                int k  = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
                long c = f3(q.nx, q.ny, i, j, k);
                ww1[c] = (uu[f3(q.nx, q.ny, i, j, k + 1)] - uu[f3(q.nx, q.ny, i, j, k - 1)]) *
                         q.hz * dzzdz[k];
            } else {
                long t2 = t - n1;
                int i   = q.i1 + (int) (t2 % (q.i2 - q.i1 + 1));
                long r  = t2 / (q.i2 - q.i1 + 1);
                int j   = (int) (r % q.ny);
                int k   = q.k1 + (int) (r / q.ny);
                long c  = f3(q.nx, q.ny, i, j, k);
                ww2[c]  = (vv[f3(q.nx, q.ny, i, j, k + 1)] - vv[f3(q.nx, q.ny, i, j, k - 1)]) *
                          q.hz * dzzdz[k];
            }
        }
    }
    __global__ void k_w3_dy(const K q, const double* ww, double* ww3, const double* dyydy) {
        long n = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i  = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r = t / (q.i2 - q.i1 + 1);
            int j  = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k  = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long c = f3(q.nx, q.ny, i, j, k);
            ww3[c] = (ww[f3(q.nx, q.ny, i, j + 1, k)] - ww[f3(q.nx, q.ny, i, j - 1, k)]) * q.hy *
                     dyydy[j];
        }
    }
    __global__ void k_ft_heat3c(const K q, const double* ro, double* ft, const double* ww1,
                                const double* ww2, const double* ww3) {
        const double tmp = q.ocv * q.ore;
        long n           = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i  = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r = t / (q.i2 - q.i1 + 1);
            int j  = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k  = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long c = f3(q.nx, q.ny, i, j, k);
            ft[c] += (ww1[c] * ww1[c] + ww2[c] * ww2[c] + ww3[c] * ww3[c] + 2.0 * ww2[c] * ww3[c]) *
                     ro[c] * tmp;
        }
    }
    // FW += C13*ORE*((dWW1/dx)h + (dWW2/dy)h)  using current WW1 (=dU/dz), WW2(=dV/dz)
    __global__ void k_fw_cross(const K q, double* fw, const double* ww1, const double* ww2,
                               const double* dxxdx, const double* dyydy) {
        const double tmpc = q.c13 * q.ore;
        long n            = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i  = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r = t / (q.i2 - q.i1 + 1);
            int j  = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k  = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long c = f3(q.nx, q.ny, i, j, k);
            fw[c] += tmpc * ((ww1[f3(q.nx, q.ny, i + 1, j, k)] - ww1[f3(q.nx, q.ny, i - 1, j, k)]) *
                                 q.hx * dxxdx[i] +
                             (ww2[f3(q.nx, q.ny, i, j + 1, k)] - ww2[f3(q.nx, q.ny, i, j - 1, k)]) *
                                 q.hy * dyydy[j]);
        }
    }
    __global__ void k_w3_dx_2(const K q, const double* ww, double* ww3, const double* dxxdx) {
        long n = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i  = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r = t / (q.i2 - q.i1 + 1);
            int j  = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k  = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long c = f3(q.nx, q.ny, i, j, k);
            ww3[c] = (ww[f3(q.nx, q.ny, i + 1, j, k)] - ww[f3(q.nx, q.ny, i - 1, j, k)]) * q.hx *
                     dxxdx[i];
        }
    }
    __global__ void k_ft_heat3d(const K q, const double* ro, double* ft, const double* ww1,
                                const double* ww3) {
        const double tmp = q.ocv * q.ore;
        long n           = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i  = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r = t / (q.i2 - q.i1 + 1);
            int j  = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k  = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long c = f3(q.nx, q.ny, i, j, k);
            ft[c] += 2.0 * ww1[c] * ww3[c] * ro[c] * tmp;
        }
    }
    // Rotation.
    __global__ void k_rot(const K q, double* fu, double* fv, double* fw, const double* ru,
                          const double* rv, const double* rw) {
        long n = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i  = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r = t / (q.i2 - q.i1 + 1);
            int j  = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k  = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long c = f3(q.nx, q.ny, i, j, k);
            fu[c] += q.omz * rv[c];
            fv[c] += -q.omz * ru[c] + q.omx * rw[c];
            fw[c] += -q.omx * rv[c];
        }
    }
    // ---------------------------------------------------------------------------
    // STEP pointwise kernels and reductions
    // ---------------------------------------------------------------------------
    __global__ void k_step_uu(const K q, const double* ru, const double* rv, const double* rw,
                              const double* ro, const double* tt, double* uu) {
        long n = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i  = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r = t / (q.i2 - q.i1 + 1);
            int j  = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k  = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long c = f3(q.nx, q.ny, i, j, k);
            double v2 =
                (ru[c] * ru[c] + rv[c] * rv[c] + rw[c] * rw[c]) * (1.0 / ro[c]) * (1.0 / ro[c]);
            // fast-mode UU = v2 + gamma*T + 2*sqrt(v2*gamma*T)
            uu[c] = v2 + q.gamma * tt[c] + 2.0 * sqrt(v2 * (q.gamma * tt[c]));
        }
    }
    // Per-cell timestep limits: WW1 (=dd/sqrt(uu)), WW2, WW3.
    __global__ void k_step_dt(const K q, const double* ro, const double* uu, const double* rkapa,
                              const double* ddx, const double* ddy, const double* ddz, double* ww1,
                              double* ww2, double* ww3) {
        long n = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i     = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r    = t / (q.i2 - q.i1 + 1);
            int j     = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k     = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long c    = f3(q.nx, q.ny, i, j, k);
            double dd = fmin(fmin(ddx[i], ddy[j]), ddz[k]);
            ww1[c]    = dd / sqrt(uu[c]);
            ww2[c]    = 0.5 * dd * dd * q.repr * q.cv * ro[c] / rkapa[k];
            ww3[c]    = 0.375 * dd * dd * q.re * ro[c];
        }
    }
    // Copy Z* = physical state (6 primaries; interior).
    __global__ void k_save_state(const K q, const double* ru, const double* rv, const double* rw,
                                 const double* ro, const double* tt, double* zru, double* zrv,
                                 double* zrw, double* zro, double* ztt) {
        long n = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i  = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r = t / (q.i2 - q.i1 + 1);
            int j  = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k  = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long c = f3(q.nx, q.ny, i, j, k);
            zru[c] = ru[c];
            zrv[c] = rv[c];
            zrw[c] = rw[c];
            zro[c] = ro[c];
            ztt[c] = tt[c];
        }
    }
    // update_prim: X = Z + coef*F  (5 primaries, interior)
    __global__ void k_update_prim(const K q, double coef, const double* zru, const double* zrv,
                                  const double* zrw, const double* zro, const double* ztt,
                                  const double* fu, const double* fv, const double* fw,
                                  const double* fr, const double* ft, double* ru, double* rv,
                                  double* rw, double* ro, double* tt) {
        long n = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i  = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r = t / (q.i2 - q.i1 + 1);
            int j  = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k  = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long c = f3(q.nx, q.ny, i, j, k);
            ru[c]  = zru[c] + coef * fu[c];
            rv[c]  = zrv[c] + coef * fv[c];
            rw[c]  = zrw[c] + coef * fw[c];
            ro[c]  = zro[c] + coef * fr[c];
            tt[c]  = ztt[c] + coef * ft[c];
        }
    }
    // mix_state: Z = X + coef*F (5 primaries, interior)
    __global__ void k_mix_state(const K q, double coef, const double* ru, const double* rv,
                                const double* rw, const double* ro, const double* tt,
                                const double* fu, const double* fv, const double* fw,
                                const double* fr, const double* ft, double* zru, double* zrv,
                                double* zrw, double* zro, double* ztt) {
        long n = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1) * (q.k2 - q.k1 + 1);
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i  = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            long r = t / (q.i2 - q.i1 + 1);
            int j  = q.j1 + (int) (r % (q.j2 - q.j1 + 1));
            int k  = q.k1 + (int) (r / (q.j2 - q.j1 + 1));
            long c = f3(q.nx, q.ny, i, j, k);
            zru[c] = ru[c] + coef * fu[c];
            zrv[c] = rv[c] + coef * fv[c];
            zrw[c] = rw[c] + coef * fw[c];
            zro[c] = ro[c] + coef * fr[c];
            ztt[c] = tt[c] + coef * ft[c];
        }
    }
    __global__ void k_zero_fill(double* p, long n) {
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x)
            p[t] = 0.0;
    }
    __global__ void k_sum(double* p, long n, double* out) {
        double s = 0.0;
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x)
            s += p[t];
        if (s != 0.0) atomicAdd(out, s);
    }

    // ---------------------------------------------------------------------------
    // Boundary conditions (BCON).  Constants (CLN, DZ) are computed on the host
    // with exact single-precision semantics and passed in.
    // ---------------------------------------------------------------------------
    __global__ void k_bcon(const K q, const double* exx, const double* wyy, const double* dzzdz,
                           const double* rkapa, double* ru, double* rv, double* rw, double* ro,
                           double* tt, double c13, double c23, double c43, double theta, double tb,
                           double tpr, double cln, double dz, int plume, int itcon, int izcon,
                           int top) {
        long n = (long) (q.i2 - q.i1 + 1) * (q.j2 - q.j1 + 1);
        int k  = top ? q.k1 : q.k2;
        // DZ = 1.0E00/FLOAT(NPZ-1)/DZZDZ(k)  (the 1/(NPZ-1) part is passed in).
        double dzloc = dz / dzzdz[k];
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            int i  = q.i1 + (int) (t % (q.i2 - q.i1 + 1));
            int j  = q.j1 + (int) (t / (q.i2 - q.i1 + 1));
            long c = f3(q.nx, q.ny, i, j, k);
            if (top) {
                long cp = f3(q.nx, q.ny, i, j, k + 1), cp2 = f3(q.nx, q.ny, i, j, k + 2);
                ru[c] = ro[c] * (c43 * ru[cp] / ro[cp] - c13 * ru[cp2] / ro[cp2]);
                rv[c] = ro[c] * (c43 * rv[cp] / ro[cp] - c13 * rv[cp2] / ro[cp2]);
                rw[c] = 0.0;
                if (!plume) {
                    if (itcon == 1)
                        tt[c] = c43 * tt[cp] - c13 * tt[cp2] - c23 * dzloc * theta;
                    else
                        tt[c] = 1.0;
                } else if (itcon == 0) {
                    double e1 = exp(cln * ((exx[i] - q.xp) * (exx[i] - q.xp)));
                    double e2 = exp(cln * ((wyy[j] - q.yp) * (wyy[j] - q.yp)));
                    tt[c]     = 1.0 - (1.0 - tpr) * e1 / q.hh * e2 / q.hh;
                } else {
                    double e1 = exp(cln * ((exx[i] - q.xp) * (exx[i] - q.xp)));
                    double e2 = exp(cln * ((wyy[j] - q.yp) * (wyy[j] - q.yp)));
                    tt[c]     = c43 * tt[cp] - c13 * tt[cp2] -
                                c23 * dzloc * (theta - (theta - tpr) * e1 / q.hh * e2 / q.hh);
                }
            } else {
                long cm = f3(q.nx, q.ny, i, j, k - 1), cm2 = f3(q.nx, q.ny, i, j, k - 2);
                ru[c] = ro[c] * (c43 * ru[cm] / ro[cm] - c13 * ru[cm2] / ro[cm2]);
                rv[c] = ro[c] * (c43 * rv[cm] / ro[cm] - c13 * rv[cm2] / ro[cm2]);
                rw[c] = 0.0;
                if (izcon == 0)
                    tt[c] = tb;
                else if (izcon == 1)
                    tt[c] = c43 * tt[cm] - c13 * tt[cm2] + c23 * dzloc * theta / rkapa[k];
            }
        }
    }
    // ---------------------------------------------------------------------------
    // Halo exchange helpers on device (for CUDA-aware MPI).
    // ---------------------------------------------------------------------------
    // A vertical slab is contiguous; y/x slabs are strided and packed here.
    __global__ void k_pack_y(double* dst, const double* src, int nx, int ny, int j0, int cnt,
                             int nz) {
        long n = (long) nx * cnt * nz;
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            long xn = t % nx;
            long r  = t / nx;
            int jj  = (int) (r % cnt);
            int z   = (int) (r / cnt);
            dst[t]  = src[(long) z * nx * ny + (long) (j0 + jj) * nx + xn];
        }
    }
    __global__ void k_unpack_y(double* dst, const double* src, int nx, int ny, int j0, int cnt,
                               int nz) {
        long n = (long) nx * cnt * nz;
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            long xn                                              = t % nx;
            long r                                               = t / nx;
            int jj                                               = (int) (r % cnt);
            int z                                                = (int) (r / cnt);
            dst[(long) z * nx * ny + (long) (j0 + jj) * nx + xn] = src[t];
        }
    }
    __global__ void k_pack_x(double* dst, const double* src, int nx, int ny, int i0, int cnt,
                             int nz) {
        long n = (long) cnt * ny * nz;
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            long xn = t % cnt;
            long r  = t / cnt;
            int y   = (int) (r % ny);
            int z   = (int) (r / ny);
            dst[t]  = src[(long) z * nx * ny + (long) y * nx + (i0 + xn)];
        }
    }
    __global__ void k_unpack_x(double* dst, const double* src, int nx, int ny, int i0, int cnt,
                               int nz) {
        long n = (long) cnt * ny * nz;
        for (long t = blockIdx.x * (long) blockDim.x + threadIdx.x; t < n;
             t += (long) blockDim.x * gridDim.x) {
            long xn                                             = t % cnt;
            long r                                              = t / cnt;
            int y                                               = (int) (r % ny);
            int z                                               = (int) (r / ny);
            dst[(long) z * nx * ny + (long) y * nx + (i0 + xn)] = src[t];
        }
    }

    // Concrete wrappers over the templated viscous kernel (cross-TU launchable).
    __global__ void k_viscous_fu(const K q, const double* uu, const double* vv, const double* ww,
                                 double* fu, double* fv, double* fw, const double* dxxdx,
                                 const double* d2xxdx2, const double* dyydy, const double* d2yydy2,
                                 const double* dzzdz, const double* d2zzdz2) {
        k_viscous<0>(q, uu, vv, ww, fu, fv, fw, dxxdx, d2xxdx2, dyydy, d2yydy2, dzzdz, d2zzdz2);
    }
    __global__ void k_viscous_fv(const K q, const double* uu, const double* vv, const double* ww,
                                 double* fu, double* fv, double* fw, const double* dxxdx,
                                 const double* d2xxdx2, const double* dyydy, const double* d2yydy2,
                                 const double* dzzdz, const double* d2zzdz2) {
        k_viscous<1>(q, uu, vv, ww, fu, fv, fw, dxxdx, d2xxdx2, dyydy, d2yydy2, dzzdz, d2zzdz2);
    }
    __global__ void k_viscous_fw(const K q, const double* uu, const double* vv, const double* ww,
                                 double* fu, double* fv, double* fw, const double* dxxdx,
                                 const double* d2xxdx2, const double* dyydy, const double* d2yydy2,
                                 const double* dzzdz, const double* d2zzdz2) {
        k_viscous<2>(q, uu, vv, ww, fu, fv, fw, dxxdx, d2xxdx2, dyydy, d2yydy2, dzzdz, d2zzdz2);
    }
}  // namespace r3d
