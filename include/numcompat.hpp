// numcompat.hpp - numeric compatibility helpers for bit-exact replication
// of the original Fortran.
//
// The original code was written for IMPLICIT REAL*8(A-H,O-Z), but its
// literals are single-precision (E00 suffix) and its integer->real casts use
// FLOAT() which returns DEFAULT REAL (kind 4).  Consequently expressions in
// which every operand is single-kind (e.g. C13 = 1.0E00/3.0E00, or
// ORX = 1.0E00/FLOAT(NX-IX-1)) are evaluated in single precision and the
// *rounded single result* is widened into the REAL*8 variable.
//
// To reproduce those values exactly we compute the same expressions in
// 'float' and widen.  f4() marks such a value.
#pragma once

namespace r3d {

inline double f4(double x) { return (double)(float)x; }

// All-operands-single integer-to-real conversions (FLOAT(j) * orx pattern):
// 1.0E00/FLOAT(n)  == f4(1.0f / (float)n)
inline double f4_over_int(double num, long den) {
  return f4((float)num / (float)den);
}

}  // namespace r3d
