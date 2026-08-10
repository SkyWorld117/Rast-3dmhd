// communicate.cpp - halo exchange port (COMM_MPI / COMMUNICATE).
#include "communicate.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

#include <mpi.h>

namespace r3d {

namespace {

// --- staging helpers for strided slabs in x-major arrays. --------------
// A slab of the form {k in [k0,k0+c)}, {j in [j0,j0+c)} or {i in [i0,i0+c)}
// for all other coordinates is contiguous only along the fastest
// (x) direction; pack/unpack make it contiguous for MPI.

// y slab {j in [j0, j0+cnt)} for all i,k      -> buf[nx*cnt*nz]
void pack_y(const Field& f, int j0, int cnt, double* buf) {
  const int nx = f.nx(), ny = f.ny(), nz = f.nz();
  double* out = buf;
  for (int k = 0; k < nz; ++k) {
    const double* src = f.data() + (long)k * nx * ny + (long)j0 * nx;
    std::memcpy(out, src, (size_t)nx * cnt * sizeof(double));
    out += (long)nx * cnt;
  }
}
void unpack_y(Field& f, int j0, int cnt, const double* buf) {
  const int nx = f.nx(), ny = f.ny(), nz = f.nz();
  const double* in = buf;
  for (int k = 0; k < nz; ++k) {
    double* dst = f.data() + (long)k * nx * ny + (long)j0 * nx;
    std::memcpy(dst, in, (size_t)nx * cnt * sizeof(double));
    in += (long)nx * cnt;
  }
}

// x slab {i in [i0, i0+cnt)} for all j,k       -> buf[cnt*ny*nz]
void pack_x(const Field& f, int i0, int cnt, double* buf) {
  const int nx = f.nx(), ny = f.ny(), nz = f.nz();
  double* out = buf;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      std::memcpy(out, f.data() + (long)k * nx * ny + (long)j * nx + i0,
                  (size_t)cnt * sizeof(double));
      out += cnt;
    }
  }
}
void unpack_x(Field& f, int i0, int cnt, const double* buf) {
  const int nx = f.nx(), ny = f.ny(), nz = f.nz();
  const double* in = buf;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      std::memcpy(f.data() + (long)k * nx * ny + (long)j * nx + i0, in,
                  (size_t)cnt * sizeof(double));
      in += cnt;
    }
  }
}

void die(const char* msg) {
  fprintf(stderr, "rast-3dmhd: %s\n", msg);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

}  // namespace

void comm_mpi(MPI_Comm comm, const Params& p, const Topology& t, Field& f) {
  const int nx = f.nx(), ny = f.ny(), nz = f.nz();
  const int iyh = p.iy / 2;
  const int ilaph = p.ilap / 2;
  if (p.ix / 2 < 1 || iyh < 1 || ilaph < 1) {
    die("communicate: IX/IY/ILAP must be at least 2");
  }

  // ------------------------------------------------------------------
  // Vertical communication (z): NX*NY*ILAP/2-thick slabs, contiguous.
  // ------------------------------------------------------------------
  const int vcount = nx * ny * ilaph;
  if (t.npez > 1) {
    const int tag1 = 100, tag2 = 200;
    if (t.mypez == 0) {
      // Send lower inner slab down; receive lower ghost from below.
      MPI_Send(f.data() + (long)(nz - p.ilap) * nx * ny, vcount, MPI_DOUBLE,
               t.mype + t.npey, tag1, comm);
      MPI_Recv(f.data() + (long)(nz - ilaph) * nx * ny, vcount, MPI_DOUBLE,
               t.mype + t.npey, tag2, comm, MPI_STATUS_IGNORE);
    } else if (t.mypez == t.npez - 1) {
      MPI_Recv(f.data(), vcount, MPI_DOUBLE, t.mype - t.npey, tag1, comm,
               MPI_STATUS_IGNORE);
      MPI_Send(f.data() + (long)ilaph * nx * ny, vcount, MPI_DOUBLE,
               t.mype - t.npey, tag2, comm);
    } else {
      MPI_Sendrecv(f.data() + (long)(nz - p.ilap) * nx * ny, vcount, MPI_DOUBLE,
                   t.mype + t.npey, tag1, f.data(), vcount, MPI_DOUBLE,
                   t.mype - t.npey, tag1, comm, MPI_STATUS_IGNORE);
      MPI_Sendrecv(f.data() + (long)ilaph * nx * ny, vcount, MPI_DOUBLE,
                   t.mype - t.npey, tag2,
                   f.data() + (long)(nz - ilaph) * nx * ny, vcount, MPI_DOUBLE,
                   t.mype + t.npey, tag2, comm, MPI_STATUS_IGNORE);
    }
  }

  // ------------------------------------------------------------------
  // Horizontal communication in y: IY/2-thick slabs (strided -> pack).
  // ------------------------------------------------------------------
  const int ycount = nx * iyh * nz;
  std::vector<double> sendbuf(ycount), recvbuf(ycount);
  if (t.npey > 1) {
    const int tag3 = 300, tag4 = 400;
    if (t.mypey == 0) {
      pack_y(f, ny - p.iy, iyh, sendbuf.data());
      MPI_Send(sendbuf.data(), ycount, MPI_DOUBLE, t.mype + 1, tag3, comm);
      MPI_Recv(recvbuf.data(), ycount, MPI_DOUBLE, t.mype + 1, tag4, comm,
               MPI_STATUS_IGNORE);
      unpack_y(f, ny - iyh, iyh, recvbuf.data());
    } else if (t.mypey == t.npey - 1) {
      MPI_Recv(recvbuf.data(), ycount, MPI_DOUBLE, t.mype - 1, tag3, comm,
               MPI_STATUS_IGNORE);
      unpack_y(f, 0, iyh, recvbuf.data());
      pack_y(f, iyh, iyh, sendbuf.data());
      MPI_Send(sendbuf.data(), ycount, MPI_DOUBLE, t.mype - 1, tag4, comm);
    } else {
      pack_y(f, ny - p.iy, iyh, sendbuf.data());
      MPI_Sendrecv(sendbuf.data(), ycount, MPI_DOUBLE, t.mype + 1, tag3,
                   recvbuf.data(), ycount, MPI_DOUBLE, t.mype - 1, tag3, comm,
                   MPI_STATUS_IGNORE);
      unpack_y(f, 0, iyh, recvbuf.data());
      pack_y(f, iyh, iyh, sendbuf.data());
      MPI_Sendrecv(sendbuf.data(), ycount, MPI_DOUBLE, t.mype - 1, tag4,
                   recvbuf.data(), ycount, MPI_DOUBLE, t.mype + 1, tag4, comm,
                   MPI_STATUS_IGNORE);
      unpack_y(f, ny - iyh, iyh, recvbuf.data());
    }
    // Periodic wrap in y (tags 500/600): edge ranks exchange with the
    // ranks at the opposite side of the y row.
    const int tag5 = 500, tag6 = 600;
    if (t.mypey == 0) {
      pack_y(f, iyh, iyh, sendbuf.data());
      MPI_Send(sendbuf.data(), ycount, MPI_DOUBLE, t.mype + t.npey - 1, tag5,
               comm);
      MPI_Recv(recvbuf.data(), ycount, MPI_DOUBLE, t.mype + t.npey - 1, tag6,
               comm, MPI_STATUS_IGNORE);
      unpack_y(f, 0, iyh, recvbuf.data());
    } else if (t.mypey == t.npey - 1) {
      MPI_Recv(recvbuf.data(), ycount, MPI_DOUBLE, t.mype - t.npey + 1, tag5,
               comm, MPI_STATUS_IGNORE);
      unpack_y(f, ny - iyh, iyh, recvbuf.data());
      pack_y(f, ny - p.iy, iyh, sendbuf.data());
      MPI_Send(sendbuf.data(), ycount, MPI_DOUBLE, t.mype - t.npey + 1, tag6,
               comm);
    }
  } else {
    // Single y rank: local periodic copy.
    pack_y(f, ny - p.iy, iyh, sendbuf.data());
    unpack_y(f, 0, iyh, sendbuf.data());
    pack_y(f, iyh, iyh, sendbuf.data());
    unpack_y(f, ny - iyh, iyh, sendbuf.data());
  }

  // ------------------------------------------------------------------
  // Periodicity in x (local copy).
  //   VAR(1:IX/2,   :, :) = VAR(NX-IX+1:NX-IX/2, :, :)
  //   VAR(NX-IX/2+1:NX,:, :) = VAR(IX/2+1:IX,    :, :)
  // ------------------------------------------------------------------
  const int xcount = (p.ix / 2) * ny * nz;
  std::vector<double> xbuf(xcount);
  pack_x(f, nx - p.ix, p.ix / 2, xbuf.data());   // interior source cols
  unpack_x(f, 0, p.ix / 2, xbuf.data());
  pack_x(f, p.ix / 2, p.ix / 2, xbuf.data());
  unpack_x(f, nx - p.ix / 2, p.ix / 2, xbuf.data());
}

void communicate(MPI_Comm comm, const Params& p, const Topology& t, SimState& s) {
  comm_mpi(comm, p, t, s.ru);
  comm_mpi(comm, p, t, s.rv);
  comm_mpi(comm, p, t, s.rw);
  comm_mpi(comm, p, t, s.tt);
  comm_mpi(comm, p, t, s.ro);
  if (p.lmag) {
    comm_mpi(comm, p, t, s.bx);
    comm_mpi(comm, p, t, s.by);
    comm_mpi(comm, p, t, s.bz);
  }
}

void horizontal_mean(MPI_Comm comm, const Params& p, const Topology& t,
                     const SimState& s, const Field& f, double* varm) {
  const int nx = f.nx(), ny = f.ny(), nz = f.nz();
  std::vector<double> wwz(nz, 0.0);
  // Interior x/y cells: i in [ix/2, nx-ix/2), j in [iy/2, ny-iy/2).
  // NGRID=0 path only (the spline branch would terminate in the original).
  if (p.ngrid != 0) {
    die("horizontal_mean: only NGRID=0 is supported (spline path unused)");
  }
  for (int k = 0; k < nz; ++k) {
    double sum = 0.0;
    for (int j = p.iy / 2; j < ny - p.iy / 2; ++j)
      for (int i = p.ix / 2; i < nx - p.ix / 2; ++i)
        sum += f.flat((long)k * nx * ny + (long)j * nx + i);
    wwz[k] = sum / (double)(p.npx * p.npy);
  }
  if (t.npey == 1) {
    std::copy(wwz.begin(), wwz.end(), varm);
  } else {
    // Reduce across y ranks; rank mypey==0 sums the per-rank partial means,
    // then broadcasts the result (mirrors HORIZONTAL_MEAN).
    std::vector<double> acc(nz, 0.0);
    std::vector<double> tmp(nz);
    if (t.mypey == 0) {
      std::copy(wwz.begin(), wwz.end(), acc.begin());
      for (int ipe = 1; ipe < t.npey; ++ipe) {
        MPI_Recv(tmp.data(), nz, MPI_DOUBLE, t.mype + ipe, 100, comm,
                 MPI_STATUS_IGNORE);
        for (int k = 0; k < nz; ++k) acc[k] += tmp[k];
      }
      for (int ipe = 1; ipe < t.npey; ++ipe)
        MPI_Send(acc.data(), nz, MPI_DOUBLE, t.mype + ipe, 200, comm);
    } else {
      MPI_Send(wwz.data(), nz, MPI_DOUBLE, t.mypez * t.npey, 100, comm);
      MPI_Recv(acc.data(), nz, MPI_DOUBLE, t.mypez * t.npey, 200, comm,
               MPI_STATUS_IGNORE);
    }
    std::copy(acc.begin(), acc.end(), varm);
  }
}

}  // namespace r3d
