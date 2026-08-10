# Rast-3dmhd C++/CUDA rewrite.
#
# Targets:
#   make            build bin/3dmhd (CPU backend by default)
#   make BACKEND=gpu   build with CUDA kernels (requires CUDA toolkit + nvcc)
#   make lib        build the static library only
#   make test       build and run the GoogleTest suite
#   make install    install the binary into $(PREFIX)/bin, the static library
#                   into $(PREFIX)/lib and the headers into $(PREFIX)/include
#   make uninstall  remove the installed files
#   make clean      remove build products
#
# ------------------------------------------------------------------------------
# Variables.  Everything below is overridable on the command line.  The
# defaults are the ones that reproduce the bit-identical (vs the Fortran
# golden) solutions; overriding them - e.g. a CXXFLAGS without -fmad=false on
# the GPU backend - can change the numerics or even make the run unstable,
# which is entirely up to the caller.
#
#   BACKEND     cpu | gpu                        (default cpu)
#   CXX         MPI C++ compiler.  MPI is always enabled, so this is *the*
#               host compiler - and, for the GPU build, the source of the MPI
#               include/link flags (override MPI_CFLAGS / MPI_LIBS if your CXX
#               is not an MPI wrapper).          (default mpicxx)
#   NVCC        CUDA compiler                    (default $(CUDA_HOME)/bin/nvcc)
#   NVCC_ARCH   CUDA compute capability          (default sm_90, H100/H200)
#   CXXFLAGS    host (CPU) compile flags         (default -O2 -g -std=c++17 ...
#               -Wall -Wextra; -Iinclude and the dump-mode define are added
#               regardless so the build never breaks from a flag override)
#   NVCCFLAGS   user-tunable device flags (default -O2 -g -std=c++17
#               -fmad=false; -fmad=false is what disables FMA contraction to
#               match the host.  Structural flags -rdc=true/--cudart=static/
#               -x cu/-arch are added by the backend regardless, so overriding
#               NVCCFLAGS changes numerics/perf but not the build model)
#   LDFLAGS     extra linker flags               (default empty)
#   LDLIBS      extra libraries to link          (default -lm)
#   DUMPMODE    periodic | final | none          (compile-time gstate policy)
#   WITH_TESTS  0|1             build/run the GoogleTest suite (default 0;
#               it requires GoogleTest, which is *not* needed for the library
#               or the binary, so it stays disabled unless asked for)
#   GTEST_PREFIX / GTEST_INC / GTEST_LIB   GoogleTest prefix / include flags /
#               static libs for the tests (defaults point at the Kez system
#               cellar; the package manager overrides GTEST_PREFIX)
#   PREFIX / DESTDIR        install destinations (default ./prefix / empty)
# ------------------------------------------------------------------------------
BACKEND   = cpu
BUILD    ?= build
BINDIR    = bin
OBJDIR    = $(BUILD)/obj
LIBDIR    = $(BUILD)/lib

PREFIX   ?= $(CURDIR)/prefix
DESTDIR  ?=

# --- toolchain (all overridable) ---------------------------------------------
CXX        = mpicxx
NVCC       = $(or $(CUDA_HOME)/bin/nvcc,/usr/local/cuda/bin/nvcc)
NVCC_ARCH  = sm_90

# -ffp-contract=off is the host side of the bit-exact contract (mirrors the
# GPU's -fmad=false): without it, an -O3/-march=native/-mfma override turns on
# GCC's default FMA contraction and the golden match fails.
# An explicit-but-empty override (e.g. a package manager passing NVCCFLAGS="")
# falls back to the bit-exact default too.
CXXFLAGS ?=
ifeq ($(strip $(CXXFLAGS)),)
  CXXFLAGS := -O2 -g -std=c++17 -Wall -Wextra -ffp-contract=off
endif

# NVCCFLAGS holds the user-tunable device flags only (opt level, standard,
# FMA contraction); the structural flags (-rdc=true, --cudart=static, -x cu,
# -arch) are appended by the backend so an override cannot break the build.
NVCCFLAGS ?=
ifeq ($(strip $(NVCCFLAGS)),)
  NVCCFLAGS := -O2 -g -std=c++17 -fmad=false
endif
LDFLAGS   ?=
LDLIBS    ?= -lm

# The GoogleTest suite is optional (WITH_TESTS=1); the base build never needs
# it.  GTEST_PREFIX is the one knob a package manager sets; the flags derive
# from it (lib64 because the system cellar keeps gtest there).
WITH_TESTS    ?= 0
GTEST_PREFIX ?= /scratch/zyi/.kez/env/system
GTEST_INC    ?= -I$(GTEST_PREFIX)/include
GTEST_LIB    ?= $(GTEST_PREFIX)/lib64/libgtest.a $(GTEST_PREFIX)/lib64/libgtest_main.a
GTEST_LIB_NM ?= $(filter-out %gtest_main.a,$(GTEST_LIB))

# --- compile-time dump mode (see src/main.cpp) --------------------------------
#   periodic - gstate dumps every NSTEP0 steps plus the final step (and start)
#   final    - only the final step's dump (and gstate.start)
#   none     - no gstate dump files at all (per-step progress is still logged)
DUMPMODE ?= periodic
DUMP_FLAG := -DR3D_DUMP_MODE=2
ifeq ($(DUMPMODE),final)
  DUMP_FLAG := -DR3D_DUMP_MODE=1
else ifeq ($(DUMPMODE),none)
  DUMP_FLAG := -DR3D_DUMP_MODE=0
else ifneq ($(DUMPMODE),periodic)
  $(error DUMPMODE must be one of: periodic, final, none)
endif

# ------------------------------------------------------------------------------
# Backends.  LIB/*.cpp are built for both; the GPU backend additionally builds
# the *.cu kernels, compiles every TU with nvcc (the driver and library call
# CUDA), and links relocatable-object (rdc) objects directly rather than via
# the archive (nvcc device-linking from an archive is unreliable).
# ------------------------------------------------------------------------------
LIB_SRCS := $(filter-out src/main.cpp,$(wildcard src/*.cpp))
GPU_SRCS := $(wildcard src/*.cu)
LIB_OBJS := $(patsubst src/%.cpp,$(OBJDIR)/%.o,$(LIB_SRCS))
GPU_OBJS := $(patsubst src/%.cu,$(OBJDIR)/%.o,$(GPU_SRCS))

ifeq ($(BACKEND),cpu)
  LIB_SRCS := $(filter-out src/gpu.cpp,$(LIB_SRCS))
  LIB_OBJS := $(patsubst src/%.cpp,$(OBJDIR)/%.o,$(LIB_SRCS))
  GPU_OBJS :=
  CC       := $(CXX)
  COMPILE  := $(CC) $(CXXFLAGS) -MMD -MP -Iinclude $(DUMP_FLAG)
  LINK     := $(CC) $(CXXFLAGS) $(LDFLAGS)
  LINK_OBJS := $(LIBDIR)/librast3dmhd.a

else ifeq ($(BACKEND),gpu)
  # MPI flags come from the MPI wrapper; nvcc needs them passed explicitly.
  MPI_CFLAGS ?= $(shell $(CXX) --showme:compile 2>/dev/null)
  MPI_LIBS   ?= $(shell $(CXX) --showme:link 2>/dev/null)
  NVCC_MPI_LIBS := $(shell echo '$(MPI_LIBS)' | sed 's/-Wl,\([^ ]*\)/-Xlinker \1/g')
  CC       := $(NVCC)
  COMPILE  := $(CC) $(NVCCFLAGS) -x cu -rdc=true --cudart=static \
              -arch=$(NVCC_ARCH) $(MPI_CFLAGS) -DR3D_HAVE_GPU -MMD -MP \
              -Iinclude $(DUMP_FLAG)
  LINK     := $(CC) $(NVCCFLAGS) $(LDFLAGS) -rdc=true --cudart=static \
              -arch=$(NVCC_ARCH) $(NVCC_MPI_LIBS) -lcudart
  LINK_OBJS := $(LIB_OBJS) $(GPU_OBJS)

else
  $(error BACKEND must be one of: cpu, gpu)
endif

# ------------------------------------------------------------------------------
# Rules.
# ------------------------------------------------------------------------------
_ := $(shell mkdir -p $(OBJDIR) $(BINDIR) $(LIBDIR))

all: $(BINDIR)/3dmhd

lib: $(LIBDIR)/librast3dmhd.a

$(BINDIR)/3dmhd: $(OBJDIR)/main.o $(LIBDIR)/librast3dmhd.a
	$(LINK) -o $@ $(OBJDIR)/main.o $(LINK_OBJS) $(LDLIBS)

$(LIBDIR)/librast3dmhd.a: $(LIB_OBJS) $(GPU_OBJS)
	ar rcs $@ $^

$(OBJDIR)/%.o: src/%.cpp
	$(COMPILE) -c -o $@ $<

$(OBJDIR)/%.o: src/%.cu
	$(COMPILE) -c -o $@ $<

-include $(wildcard $(OBJDIR)/*.d)

# ------------------------------------------------------------------------------
# Tests (WITH_TESTS=1).  GoogleTest is optional: the library and binary never
# need it, so the whole suite is gated behind WITH_TESTS.  Each test links the
# library; the MPI-flavoured tests provide their own main() (they call
# MPI_Init), so they link gtest without gtest_main and are run under mpirun by
# 'make test_multi' rather than the single-process 'make test' (NPEZ >= 2).
# ------------------------------------------------------------------------------
ifeq ($(WITH_TESTS),1)
TESTS     := $(patsubst tests/%_test.cpp,%,$(wildcard tests/*_test.cpp))
TEST_BINS := $(addprefix tests/,$(addsuffix _test,$(TESTS)))
MULTI     := tests/communicate_test tests/static_test tests/step_test tests/gpu_equivalence_test
GPU_TESTS := tests/gpu_equivalence_test
SINGLE    := $(filter-out $(MULTI),$(TEST_BINS))

$(OBJDIR)/%_test.o: tests/%_test.cpp
	$(COMPILE) $(GTEST_INC) -c -o $@ $<

$(TEST_BINS): tests/%_test: $(OBJDIR)/%_test.o $(LIBDIR)/librast3dmhd.a
	$(LINK) $(GTEST_INC) -o $@ $< $(LINK_OBJS) \
	  $(if $(filter $@,$(MULTI)),$(GTEST_LIB_NM),$(GTEST_LIB)) $(LDLIBS)

test: $(SINGLE)
	@for t in $(SINGLE); do echo; echo "== $$t =="; ./$$t || exit 1; done

test_multi: $(filter-out $(GPU_TESTS),$(MULTI))
	mpirun -np 4 ./tests/communicate_test
	mpirun -np 4 ./tests/static_test
	mpirun -np 4 ./tests/step_test

test_gpu: $(GPU_TESTS)
	mpirun -np 4 ./tests/gpu_equivalence_test

else
test:
	@echo "GoogleTest suite disabled. Rebuild with: make WITH_TESTS=1 test"

test_multi:
	@echo "GoogleTest suite disabled. Rebuild with: make WITH_TESTS=1 test_multi"

test_gpu:
	@echo "GoogleTest suite disabled. Rebuild with: make WITH_TESTS=1 test_gpu"

endif

# ------------------------------------------------------------------------------
# Install.
# ------------------------------------------------------------------------------
install: $(BINDIR)/3dmhd $(LIBDIR)/librast3dmhd.a
	install -d $(DESTDIR)$(PREFIX)/bin $(DESTDIR)$(PREFIX)/lib $(DESTDIR)$(PREFIX)/include
	install -m 0755 $(BINDIR)/3dmhd      $(DESTDIR)$(PREFIX)/bin/3dmhd
	install -m 0644 $(LIBDIR)/librast3dmhd.a $(DESTDIR)$(PREFIX)/lib/librast3dmhd.a
	install -m 0644 include/*.hpp        $(DESTDIR)$(PREFIX)/include/

uninstall:
	rm -f $(DESTDIR)$(PREFIX)/bin/3dmhd
	rm -f $(DESTDIR)$(PREFIX)/lib/librast3dmhd.a
	rm -f $(DESTDIR)$(PREFIX)/include/*.hpp

clean:
	rm -rf $(OBJDIR) $(BINDIR) $(LIBDIR) $(TEST_BINS)

.PHONY: all lib test test_multi test_gpu install uninstall clean
