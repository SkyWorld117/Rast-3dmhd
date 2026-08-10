// topology_test - 2D processor-grid resolution and rank mapping.
#include <gtest/gtest.h>

#include "params.hpp"
#include "topology.hpp"

using namespace r3d;

TEST(RankXY, MappingMatchesFortran) {
  // Original Fortran: MYPEY=MOD(MYPE,NPEY), MYPEZ=MYPE/NPEY.
  EXPECT_EQ(rank_to_xy(0, 2).mypey, 0);
  EXPECT_EQ(rank_to_xy(0, 2).mypez, 0);
  EXPECT_EQ(rank_to_xy(1, 2).mypey, 1);
  EXPECT_EQ(rank_to_xy(1, 2).mypez, 0);
  EXPECT_EQ(rank_to_xy(2, 2).mypey, 0);
  EXPECT_EQ(rank_to_xy(2, 2).mypez, 1);
  EXPECT_EQ(rank_to_xy(3, 2).mypey, 1);
  EXPECT_EQ(rank_to_xy(3, 2).mypez, 1);
  // npey = 3
  EXPECT_EQ(rank_to_xy(4, 3).mypey, 1);
  EXPECT_EQ(rank_to_xy(4, 3).mypez, 1);
  EXPECT_EQ(rank_to_xy(5, 3).mypey, 2);
  EXPECT_EQ(rank_to_xy(5, 3).mypez, 1);
}

TEST(Topology, ExplicitGrid) {
  Params p;
  p.npey = 2;
  Topology t = resolve_topology(4, 3, p);
  EXPECT_EQ(t.npey, 2);
  EXPECT_EQ(t.npez, 2);
  EXPECT_EQ(t.mypey, 1);
  EXPECT_EQ(t.mypez, 1);
}

TEST(Topology, GoldenFourRankGrid) {
  Params p;
  p.npx = 16; p.npy = 16; p.npz = 32;
  p.npey = 2;
  Topology t = resolve_topology(4, 0, p);
  EXPECT_EQ(t.npey, 2);
  EXPECT_EQ(t.npez, 2);
  EXPECT_EQ(t.mypey, 0);
  EXPECT_EQ(t.mypez, 0);
  Derived d = derive(p, t.npe, t.npey, t.npez);
  EXPECT_EQ(d.nry, 8);
  EXPECT_EQ(d.nrz, 16);
  EXPECT_EQ(d.nx, 18);
  EXPECT_EQ(d.ny, 10);
  EXPECT_EQ(d.nz, 18);
}

TEST(Topology, AutoGridIsSquareish) {
  Params p;
  p.npey = 0;  // auto
  Topology t = resolve_topology(16, 0, p);
  EXPECT_EQ(t.npey * t.npez, 16);
  EXPECT_GE(t.npez, 2);
  // After resolution, np must divide p.npy and p.npz for the solver to work;
  // here the default 504 is divisible by 4 (=npey) and 4 (=npez).
  EXPECT_EQ(p.npy % t.npey, 0);
  EXPECT_EQ(p.npz % t.npez, 0);
}

TEST(Topology, NpezMustBeAtLeastTwo) {
  Params p;
  p.npey = 1;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(resolve_topology(1, 0, p), ::testing::ExitedWithCode(1),
              "NPEZ >= 2");
}

TEST(Topology, IndivisibleGridIsRejected) {
  Params p;
  p.npy = 7;
  p.npey = 4;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(resolve_topology(8, 0, p), ::testing::ExitedWithCode(1),
              "not divisible");
}
