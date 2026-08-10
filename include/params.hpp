// params.hpp - runtime replacement for the compile-time PARAMETER statements
// in the original 3dmhdparam.f, plus the IPAR/PAR arrays from 3dmhdset.f.
//
// The original code hardcoded the domain geometry, processor grid and all
// physical/control parameters in a Fortran PARAMETER file and a SETUP
// routine.  Here they are a plain struct filled from the command line
// (with the exact original defaults), so a run is described entirely by
// argv and the MPI topology is NOT hardcoded.
//
// All arithmetic in the physics core is double precision, as in the
// original (IMPLICIT REAL*8).  Values in this header mirror the Fortran
// literals verbatim so the C++ port can match the golden reference.
#pragma once

#include <cstdint>

#include "numcompat.hpp"

namespace r3d {

    // ---------------------------------------------------------------------------
    // Raw user-facing parameters (1:1 with 3dmhdparam.f + IPAR/PAR).
    // ---------------------------------------------------------------------------
    struct Params {
        // --- domain geometry / stencil widths (3dmhdparam.f) ---
        int npx   = 504;   // global points in x
        int npy   = 504;   // global points in y
        int npz   = 2048;  // global points in z
        int npey  = 8;     // processors in y (npez is derived from MPI size)
        int ix    = 2;     // horizontal stencil half-width (ghost/border width)
        int iy    = 2;
        int ilap  = 2;           // vertical stencil half-width
        int ipad  = 0;           // padding (unused, kept for orthogonality)
        int ngrid = 1;           // grid-stretch function selector (0,1,2)
        int id    = 0;           // vertical variation of density diffusion
        int ncol = 0, nrow = 0;  // tube array geometry (NTUBES = NCOL*NROW)

        // --- boundary-condition selectors ---
        int ixcon = 0;  // 0 = periodic x
        int iycon = 0;
        int izcon = 0;  // lower boundary temperature (0 const T, 1 flux)
        int itcon = 1;  // upper boundary/plume condition (see code)
        int ibcon = 0;  // B-field boundary (0 = Bz = 0)

        // --- physics toggles ---
        bool lrot = false;  // rotation
        bool lmag = false;  // magnetic fields
        bool lpot = false;  // enforce div B = 0
        bool lrem = false;  // M. Rempel radiative-conduction stratification
        bool lshr = false;  // shear (horizontal shear instability setup)

        // --- run control (IPAR) ---
        int ncase  = 1;
        int ncasep = 1;
        int ntotal = 1000;
        int nstep0 = 100;
        int nstart = 0;  // 0 = new start (STATIC), else restart dump index
        int ntube  = 0;  // tube model number (initial condition)

        // --- physical parameters (PAR) ---
        // The original SETUP routine assigns these from single-precision (E00)
        // literals into the REAL*8 PAR array, so values that are not exactly
        // representable in float arrive widened:  PR, AMPT, GAMMA below.
        double re     = 2.0;              // 01 soundspeed Reynolds number
        double pr     = f4(5.0e-2f);      // 02 pseudo Prandtl number (0.05 in float)
        double theta  = 2.5e-1;           // 03 temperature gradient (exact)
        double grav   = 6.25e-1;          // 04 nondim gravitational constant (exact)
        double r_y    = 0.0;              // 05 soundspeed Rossby number
        double ang    = 0.0;              // 06 rotation axis angle from vertical
        double rm     = 0.0;              // 07 magnetic Reynolds number
        double beta   = 0.0;              // 08 plasma beta
        double pzp    = 0.0;              // 09 depth of stable-background transition
        double sigma  = 0.0;              // 10 width of transition region
        double polys  = 0.0;              // 11 polytropic index of lower stable layer
        double tp     = 2.0;              // 12 plume temperature / flux / nonuniformity
        double tc     = 0.0;              // 13 cooling/onset time scale
        double xp     = 10.0;             // 14 plume source x
        double yp     = 10.0;             // 15 plume source y
        double zp     = 0.0;              // 16 plume source z
        double xmax   = 20.0;             // 17 domain x extent
        double ymax   = 20.0;             // 18 domain y extent
        double zmax   = 40.0;             // 19 domain z extent
        double ampt   = f4(5.0e-3f);      // 20 amplitude of initial random perturbations
        double ampb   = 0.0;              // 21 magnetic layer amplitude
        double bfh    = 0.0;              // 22 magnetic layer FWHM
        double bzp    = 0.0;              // 23 magnetic layer depth
        double qfh    = 0.0;              // 53 embedded-heat-loss temporal FWHM
        double gamma  = f4(5.0f / 3.0f);  // 54 Cp/Cv (5.00E00/3.00E00 in float)
        double hh     = 1.0;              // 55 tube width / plume spatial width
        double dh     = 0.0;              // 56 horizontal density diffusion
        double dv     = 0.0;              // 57 vertical density diffusion
        double tstart = 0.0;              // 58 heat-flux relaxation start
        double toff   = 0.0;              // 59 heat-flux relaxation end scale
        double rlax   = 0.0;              // 60 heat-flux relaxation amplitude

        // --- grid-stretching parameters (PARAMETER in 3dmhdparam.f) ---
        double xx1 = -7.0, xx2 = 7.0;
        double yy1 = -7.0, yy2 = 7.0;
        double zz1 = -2.0, zz2 = 1.0e-2;
        double xa = 0.5, xb = 0.48, xc = 28.0, xd = 0.025;
        double ya = 0.5, yb = 0.48, yc = 28.0, yd = 0.025;
        double za = 0.65, zb = 1.0, zc = 5.0, zd = 0.2;

        // --- tube parameters (PAR 34..38, 51..52) ---
        double xcent = 0.0, zcent = 0.0;  // tube center
        double rmax   = 0.0;              // radius at which B -> 0
        double cmt    = 0.0;              // pitch angle parameter
        double a_tube = 0.0;              // B_phi profile parameter
        double lambda = 0.0;              // wavelength of 3d tube perturbation
        double vz0    = 0.0;              // amplitude of 3d tube perturbation

        // --- time stepping ---
        // SFF = 0.315E00 is a single-precision literal in 3dmhdparam.f.
        double sf = f4(0.315f);  // safety factor

        // --- backend ---
        enum class Backend { kCpu, kGpu };
        Backend backend = Backend::kCpu;
        int gpu_id      = 0;
    };

    // ---------------------------------------------------------------------------
    // Values derived from Params + MPI topology (3dmhd.f setup section).
    // ---------------------------------------------------------------------------
    struct Derived {
        int np;          // MPI comm size
        int npey, npez;  // resolved processor grid (npey*nyez == np)
        int nry, nrz;    // interior points per rank in y/z (NPY/NPEY, NPZ/NPEZ)
        int nx, ny, nz;  // local array extents incl. ghosts

        // grid metric factors (3dmhd.f lines 112-127)
        double c13, c23, c43;
        double gam1, gam2, gam3, zeta1, zeta2;  // Wray RK3 coefficients
        double hx, h2x, hy, h2y, hz, h2z;

        // combined / derived physical constants
        double repr;      // Re*Pr
        double ore;       // 1/Re
        double cv;        // 1/(gamma-1)
        double ocv;       // 1/cv
        double rkapst;    // (polys+1)*theta/grav (or 1 if grav==0)
        double orm;       // 1/Rm (0 if no mag)
        double obeta;     // 2/beta (0 if no mag)
        double omx, omz;  // rotation rates (0 if no rotation)
    };

    // Fill Derived from Params + a resolved 2D topology.  Assumes np == npey*npez
    // and that npy%npey==0 and npz%npez==0 (validated by the caller).
    Derived derive(const Params& p, int np, int npey, int npez);

    // ---------------------------------------------------------------------------
    // Command-line parsing.
    // ---------------------------------------------------------------------------
    // Parse argv into p.  Unknown flags are an error (the code terminates via
    // ERROR()).  Returns 0 on success.
    int parse_args(int argc, char** argv, Params& p);

    void print_help();

}  // namespace r3d
