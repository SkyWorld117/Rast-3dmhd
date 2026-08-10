// ode.hpp - Bulirsch-Stoer integration used by STATIC to build the vertical
// stratification (DERIVS / BSSTEP / MMID / RZEXTR, NR p. 560-566).
#pragma once

#include "params.hpp"

namespace r3d {

// Derivative callback signature (NV state variables).
//  zz = independent variable (computational coordinate in [0,1]).
using DerivFn = void (*)(const Params& p, double zz, const double* y, double* dydx);

// DERIVS from 3dmhdsub.f: dT/dZZ and dlnP/dZZ for the static atmosphere.
void derivs_static(const Params& p, double zz, const double* y, double* dydx);

// Bulirsch-Stoer step with monitoring; on return x is advanced, hdid is the
// step taken, hnext a suggested next step.  Caller must supply the SAVE
// buffer 'xbuf[11]' used by RZEXTR across substeps within one call.
void bsstep(const Params& p, double* y, double* dydx, int nv, double& x,
            double htry, double eps, double& hdid, double& hnext,
            DerivFn derivs, double* xbuf);

// Build the full static profile used by STATIC when LREM is false:
// T(k), R(k) for k = 0..npz-1 (k=0 is the upper boundary), with
//   T(0) = 1, R(0) = 1 and rows integrated with BSSTEP at fixed DZZ.
void static_profile(const Params& p, int npz, double* tprof, double* rprof);

}  // namespace r3d
