// fields.hpp - 3D field storage and the simulation state.
//
// Arrays are stored flat, x fastest:  element (i,j,k) lives at
//     i + nx*(j + ny*k)
// which is exactly the memory layout of the original Fortran arrays
// (NX,NY,NZ) declared in column-major order.  Consequently every loop in
// the port is a 1:1 transcription of the Fortran loops; 0-based indexing
// here corresponds to 1-based indexing in the Fortran with a -1 offset.
//
// Interior (physical) cells of a local rank domain are
//     i in [1, nx-1-1]                    (Fortran 2..NX-IX+1)
//     j in [1, ny-1-1]                    (Fortran 2..NY-IY+1)
//     k in [ilap/2, nz-ilap/2-1]          (Fortran ILAP/2+1..NZ-ILAP/2)
// where nx-ix-1 == npx (interior x points), etc.  The ghost/border layers
// are outside these ranges and are (re)filled by COMMUNICATE / BCON only
// when the stencils require them.
#pragma once

#include <cstddef>
#include <vector>

#include "params.hpp"

namespace r3d {

    // ---------------------------------------------------------------------------
    // Layout helpers.
    // ---------------------------------------------------------------------------
    struct Grid3 {
        int nx = 0, ny = 0, nz = 0;
        long idx(int i, int j, int k) const { return (long) k * nx * ny + (long) j * nx + i; }
        long size() const { return (long) nx * ny * nz; }
    };

    inline long idx3(int nx, int ny, int i, int j, int k) {
        return (long) k * nx * ny + (long) j * nx + i;
    }

    // Owning host-side 3D array.
    class Field {
       public:
        Field() = default;
        Field(const Grid3& g, double init) : g_(g), v_(g.size(), init) {}
        Field(const Grid3& g, const double* src) : g_(g), v_(src, src + g.size()) {}

        double* data() { return v_.data(); }
        const double* data() const { return v_.data(); }
        const Grid3& grid() const { return g_; }
        int nx() const { return g_.nx; }
        int ny() const { return g_.ny; }
        int nz() const { return g_.nz; }
        long size() const { return g_.size(); }

        double& at(int i, int j, int k) { return v_[g_.idx(i, j, k)]; }
        const double& at(int i, int j, int k) const { return v_[g_.idx(i, j, k)]; }
        double& flat(long n) { return v_[n]; }
        const double& flat(long n) const { return v_[n]; }

        void fill(double v) { std::fill(v_.begin(), v_.end(), v); }

        // Copy whole array.
        void copy_from(const Field& o) {
            g_ = o.g_;
            v_.assign(o.v_.begin(), o.v_.end());
        }

       private:
        Grid3 g_;
        std::vector<double> v_;
    };

    // ---------------------------------------------------------------------------
    // The full simulation state (all /BIG/ arrays + 1D metrics + scalars).
    // ---------------------------------------------------------------------------
    struct SimState {
        explicit SimState(const Derived& d) : d(d) {
            Grid3 g {d.nx, d.ny, d.nz};
            ru  = Field(g, 0.0);
            rv  = Field(g, 0.0);
            rw  = Field(g, 0.0);
            ro  = Field(g, 0.0);
            tt  = Field(g, 0.0);
            uu  = Field(g, 0.0);
            vv  = Field(g, 0.0);
            ww  = Field(g, 0.0);
            fu  = Field(g, 0.0);
            fv  = Field(g, 0.0);
            fw  = Field(g, 0.0);
            fr  = Field(g, 0.0);
            ft  = Field(g, 0.0);
            zru = Field(g, 0.0);
            zrv = Field(g, 0.0);
            zrw = Field(g, 0.0);
            zro = Field(g, 0.0);
            ztt = Field(g, 0.0);
            ww1 = Field(g, 0.0);
            ww2 = Field(g, 0.0);
            ww3 = Field(g, 0.0);
            bx  = Field(g, 0.0);
            by  = Field(g, 0.0);
            bz  = Field(g, 0.0);
            zbx = Field(g, 0.0);
            zby = Field(g, 0.0);
            zbz = Field(g, 0.0);
            exx.resize(d.nx);
            dxxdx.resize(d.nx);
            d2xxdx2.resize(d.nx);
            ddx.resize(d.nx);
            wyy.resize(d.ny);
            dyydy.resize(d.ny);
            d2yydy2.resize(d.ny);
            ddy.resize(d.ny);
            zee.resize(d.nz);
            dzzdz.resize(d.nz);
            d2zzdz2.resize(d.nz);
            ddz.resize(d.nz);
            rkapa.resize(d.nz);
            dkapa.resize(d.nz);
        }

        Derived d;

        // Primary (physical) fields.
        Field ru, rv, rw, ro, tt;
        // Derived velocities / work arrays.
        Field uu, vv, ww;
        // Fluxes (time derivatives).
        Field fu, fv, fw, fr, ft;
        // Runge-Kutta saved state.
        Field zru, zrv, zrw, zro, ztt;
        // Generic work arrays (also B-field fluxes when LMAG).
        Field ww1, ww2, ww3;
        // Magnetic fields.
        Field bx, by, bz, zbx, zby, zbz;

        // 1D grid / conductivity.
        std::vector<double> exx, dxxdx, d2xxdx2, ddx;  // length nx
        std::vector<double> wyy, dyydy, d2yydy2, ddy;  // length ny
        std::vector<double> zee, dzzdz, d2zzdz2, ddz;  // length nz
        std::vector<double> rkapa, dkapa;              // length nz

        // Global scalars carried by the driver.
        double dt = 0.0, timc = 0.0, timt = 0.0, timi = 0.0, umach = 0.0;
        double tb  = 1.0;  // lower boundary temperature (from STATIC)
        int nit    = 0;
        int ndump0 = 0;  // number of dumps emitted
    };

}  // namespace r3d
