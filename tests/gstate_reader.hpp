// gstate_reader.hpp - test helper that decodes the golden gstate files
// produced by golden/fortran/gstate_sub.f (see read_gstate.py for the
// format documentation).  Header-only.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace r3dtest {

    struct Gstate {
        int32_t nx = 0, ny = 0, nz = 0;
        int32_t mype = 0, mypey = 0, mypez = 0, nit = 0, lmag = 0;
        int32_t ixcon = 0, iycon = 0, izcon = 0, itcon = 0, ibcon = 0;
        double scalars[27] = {0};
        // 3D fields in a fixed order, flat x-fastest.
        std::vector<double> f3d[27];
        // 1D metrics: 4 x-arrays, 4 y-arrays, 6 z-arrays.
        std::vector<double> f1d[14];

        static int idx3d(const std::string& name);
        static int idx1d(const std::string& name);
        const std::vector<double>& field3d(const std::string& name) const;
        const std::vector<double>& field1d(const std::string& name) const;
    };

    inline int Gstate::idx3d(const std::string& name) {
        static const char* names[27] = {"ru",  "rv",  "rw",  "ro",  "tt",  "uu",  "vv",
                                        "ww",  "fu",  "fv",  "fw",  "fr",  "ft",  "zru",
                                        "zrv", "zrw", "zro", "ztt", "ww1", "ww2", "ww3",
                                        "bx",  "by",  "bz",  "zbx", "zby", "zbz"};
        for (int i = 0; i < 27; ++i)
            if (name == names[i]) return i;
        return -1;
    }
    inline int Gstate::idx1d(const std::string& name) {
        static const char* names[14] = {"exx",     "dxxdx",   "d2xxdx2", "ddx",  "wyy",
                                        "dyydy",   "d2yydy2", "ddy",     "zee",  "dzzdz",
                                        "d2zzdz2", "ddz",     "rkapa",   "dkapa"};
        for (int i = 0; i < 14; ++i)
            if (name == names[i]) return i;
        return -1;
    }
    inline const std::vector<double>& Gstate::field3d(const std::string& name) const {
        int k = idx3d(name);
        if (k < 0) {
            fprintf(stderr, "gstate_reader: unknown 3D field '%s'\n", name.c_str());
            exit(EXIT_FAILURE);
        }
        return f3d[k];
    }
    inline const std::vector<double>& Gstate::field1d(const std::string& name) const {
        int k = idx1d(name);
        if (k < 0) {
            fprintf(stderr, "gstate_reader: unknown 1D field '%s'\n", name.c_str());
            exit(EXIT_FAILURE);
        }
        return f1d[k];
    }

    // Read a gstate file; fatal on format errors or extra trailing bytes.
    inline Gstate read_gstate(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            fprintf(stderr, "gstate_reader: cannot open '%s'\n", path.c_str());
            exit(EXIT_FAILURE);
        }
        std::vector<char> buf((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
        size_t off = 0;
        auto mread = [&](void* out, size_t nbytes) {
            if (off + nbytes > buf.size()) {
                fprintf(stderr, "gstate_reader: truncated file '%s'\n", path.c_str());
                exit(EXIT_FAILURE);
            }
            memcpy(out, buf.data() + off, nbytes);
            off += nbytes;
        };

        int32_t h[3];
        mread(h, sizeof(h));
        if (h[0] != 1129928788 || h[1] != 27 || h[2] != 14) {
            fprintf(stderr, "gstate_reader: bad header in '%s'\n", path.c_str());
            exit(EXIT_FAILURE);
        }
        Gstate g;
        mread(&g.nx, 13 * sizeof(int32_t));  // nx,ny,nz,mype,mypey,mypez,nit,lmag,ix..ib
        // NOTE: struct field order after nz is mype,mypey,mypez,nit,lmag,ixcon...
        // The raw order is nx,ny,nz,mype,mypey,mypez,nit,lmag,ixcon,iycon,izcon,itcon,ibcon
        // which matches the struct layout above.
        double* s = g.scalars;
        mread(s, 27 * sizeof(double));
        // Field arrays (the struct fields above follow the raw order).
        int64_t n3 = (int64_t) g.nx * g.ny * g.nz;
        for (int i = 0; i < 27; ++i) {
            int64_t cnt;
            mread(&cnt, sizeof(cnt));
            if (cnt != n3) {
                fprintf(stderr, "gstate_reader: 3D field %d has count %lld\n", i, (long long) cnt);
                exit(EXIT_FAILURE);
            }
            g.f3d[i].resize(n3);
            mread(g.f3d[i].data(), n3 * sizeof(double));
        }
        for (int i = 0; i < 14; ++i) {
            int64_t cnt;
            mread(&cnt, sizeof(cnt));
            g.f1d[i].resize(cnt);
            mread(g.f1d[i].data(), cnt * sizeof(double));
        }
        int64_t end;
        mread(&end, sizeof(end));
        if (end != -1) {
            fprintf(stderr, "gstate_reader: bad endmark in '%s'\n", path.c_str());
            exit(EXIT_FAILURE);
        }
        if (off != buf.size()) {
            fprintf(stderr, "gstate_reader: %zu trailing bytes in '%s'\n", buf.size() - off,
                    path.c_str());
            exit(EXIT_FAILURE);
        }
        return g;
    }

}  // namespace r3dtest
