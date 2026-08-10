#!/usr/bin/env python3
"""Decode the golden gstate files (self-describing format from
gstate_sub.f). Layout:

  int32 magic   = 1129928788
  int32 nfields = 27, int32 n1d = 14
  int32[13]     nx ny nz mype mypey mypez nit lmag ixcon iycon izcon itcon ibcon
  f64[27]         dt timc timt timi umach re repr cv ocv ore theta grav
                rkapst orm rm obeta omx omz sf ampt dd hx h2x hy h2y hz h2z
  27 blocks     : int64 count, f64[count]   (ru..zbz, x-fastest)
  14 blocks     : int64 count, f64[count]   (1D metrics)
  int64         : -1 (endmark)
"""
import struct, os

MAGIC = 1129928788
FIELD3D = ["ru","rv","rw","ro","tt","uu","vv","ww",
           "fu","fv","fw","fr","ft",
           "zru","zrv","zrw","zro","ztt",
           "ww1","ww2","ww3",
           "bx","by","bz","zbx","zby","zbz"]
FIELD1D = ["exx","dxxdx","d2xxdx2","ddx",
           "wyy","dyydy","d2yydy2","ddy",
           "zee","dzzdz","d2zzdz2","ddz","rkapa","dkapa"]
INTKEYS = ["nx","ny","nz","mype","mypey","mypez","nit","lmag",
           "ixcon","iycon","izcon","itcon","ibcon"]
SCALKEYS = ["dt","timc","timt","timi","umach","re","repr","cv","ocv","ore",
            "theta","grav","rkapst","orm","rm","obeta","omx","omz","sf","ampt",
            "dd","hx","h2x","hy","h2y","hz","h2z"]


class DecodeError(Exception):
    pass


def read(path):
    with open(path, "rb") as f:
        data = f.read()
    off = 0

    def take(fmt):
        nonlocal off
        sz = struct.calcsize(fmt)
        if off + sz > len(data):
            raise DecodeError(f"truncated at offset {off}")
        v = struct.unpack_from(fmt, data, off)
        off += sz
        return v

    magic, nf, n1 = take("<3i")
    if magic != MAGIC:
        raise DecodeError(f"bad magic {magic}")
    if nf != len(FIELD3D) or n1 != len(FIELD1D):
        raise DecodeError(f"unexpected block counts {nf},{n1}")
    iv = take("<13i")
    sv = take("<27d")
    h = dict(zip(INTKEYS, iv))
    h["scalars"] = dict(zip(SCALKEYS, sv))
    nx, ny, nz = h["nx"], h["ny"], h["nz"]
    for name in FIELD3D:
        (cnt,) = take("<q")
        if cnt != nx * ny * nz:
            raise DecodeError(f"{name}: count {cnt} != {nx*ny*nz}")
        h[name] = list(take(f"<{cnt}d"))
    for name in FIELD1D:
        (cnt,) = take("<q")
        h[name] = list(take(f"<{cnt}d"))
    (end,) = take("<q")
    if end != -1:
        raise DecodeError(f"bad endmark {end}")
    if off != len(data):
        raise DecodeError(f"trailing bytes: read {off} of {len(data)}")
    h["_bytes_read"] = off
    return h


def summary(path, arrays=("ru","tt","ro","uu","ww1","fu","exx","wyy","zee","rkapa")):
    import math
    h = read(path)
    nn = h["nx"]*h["ny"]*h["nz"]
    print(f"{os.path.basename(path)}: n={h['nx']}x{h['ny']}x{h['nz']} "
          f"mype={h['mype']}(y={h['mypey']},z={h['mypez']}) nit={h['nit']} lmag={h['lmag']}")
    print("  scalars:", {k: round(v, 6) for k, v in h["scalars"].items()
                         if k in ("dt","timc","umach","re","repr","cv","ocv","ore",
                                  "theta","grav","rkapst","orm","obeta","sf","ampt")})
    for a in arrays:
        v = h[a]
        k = "3d" if len(v) == nn else ("nx" if len(v) == h["nx"] else
             "ny" if len(v) == h["ny"] else "nz")
        print(f"  {a:6s}[{k}] min={min(v): .9e} max={max(v): .9e}")
    return h


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        summary(p)
