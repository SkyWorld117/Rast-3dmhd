# Rast-3dmhd C++/CUDA rewrite.
#
# Targets:
#   make            build bin/3dmhd (CPU backend by default)
#   make BACKEND=gpu   build with CUDA kernels (requires CUDA toolkit + nvcc)
#   make lib        build the static library only
#   make test       build and run the GoogleTest suite
#   make clean      remove build products
#
# Overridable variables (the Kez recipe sets these):
#   CXX / MPICXX / NVCC   compilers
#   GTEST_INC / GTEST_LIB gtest include dir and libs
#   NVCC_ARCH             compute capability (default sm_90 for H200)
#   CXXFLAGS / NVCCFLAGS
BACKEND  = cpu
BUILD   ?= build
BINDIR   = bin
OBJDIR   = $(BUILD)/obj
LIBDIR   = $(BUILD)/lib

CXX      = mpicxx   # MPI code: use the MPI wrapper by default
CXXFLAGS  = -O2 -g -std=c++17 -Wall -Wextra -Iinclude
LDLIBS   ?= -lm

MPICXX   = mpicxx
LINKXX   = $(CXX)
NVCC    = $(or $(CUDA_HOME)/bin/nvcc,/usr/local/cuda/bin/nvcc)
NVCC_ARCH  = sm_90

GTEST_INC  = -I/scratch/zyi/.kez/env/system/include
GTEST_LIB  = /scratch/zyi/.kez/env/system/lib64/libgtest.a /scratch/zyi/.kez/env/system/lib64/libgtest_main.a

# --------------------------------------------------------------------------
# Sources.  Kernel files (*.cu) are only compiled for the GPU backend.
# --------------------------------------------------------------------------
LIB_SRCS := $(filter-out src/main.cpp,$(wildcard src/*.cpp))
GPU_SRCS := $(wildcard src/*.cu)
LIB_OBJS := $(patsubst src/%.cpp,$(OBJDIR)/%.o,$(LIB_SRCS))
GPU_OBJS :=

ifeq ($(BACKEND),cpu)
  LIB_SRCS := $(filter-out src/gpu.cpp,$(LIB_SRCS))
  LIB_OBJS := $(patsubst src/%.cpp,$(OBJDIR)/%.o,$(LIB_SRCS))
endif

ifeq ($(BACKEND),gpu)
  MPI_CFLAGS := $(shell $(MPICXX) --showme:compile 2>/dev/null)
  MPI_LIBS   := $(shell $(MPICXX) --showme:link 2>/dev/null)
  CXX := $(NVCC)
  CXXFLAGS := $(filter-out -Wall -Wextra,$(CXXFLAGS))
  CXXFLAGS += -x cu -rdc=true --cudart=static -arch=$(NVCC_ARCH) $(MPI_CFLAGS)
  # Bit-exact port: nvcc fuses a*b+c into FMA by default, which the host
  # compiler (plain g++, no -ffast-math) does not - disable contraction so the
  # GPU arithmetic matches the CPU/golden rounding exactly.
  CXXFLAGS += -fmad=false
  # Bit-exact port: nvcc fuses a*b+c into FMA by default, which the host
  # compiler (plain g++, no -ffast-math) does not - disable contraction so the
  # GPU arithmetic matches the CPU/golden rounding exactly.

  LINKXX := $(NVCC)
  MPI_LIBS := $(shell echo '$(MPI_LIBS)' | sed 's/-Wl,\([^ ]*\)/-Xlinker \1/g')
  LINKFLAGS += -rdc=true -arch=$(NVCC_ARCH) --cudart=static $(MPI_LIBS) -lcudart
  LINK_CXXFLAGS := $(filter-out -x cu,$(CXXFLAGS))
  LINK_LIB = $(LIB_OBJS) $(GPU_OBJS)
  GPU_OBJS := $(patsubst src/%.cu,$(OBJDIR)/%.o,$(GPU_SRCS))
endif

TESTS      := $(patsubst tests/%_test.cpp,%,$(wildcard tests/*_test.cpp))
TEST_BINS  := $(addprefix tests/,$(addsuffix _test,$(TESTS)))
_ := $(shell mkdir -p $(OBJDIR) $(BINDIR) $(LIBDIR))

all: $(BINDIR)/3dmhd

LINK_LIB ?= $(LIBDIR)/librast3dmhd.a
lib: $(LIBDIR)/librast3dmhd.a

$(BINDIR)/3dmhd: $(OBJDIR)/main.o $(LIBDIR)/librast3dmhd.a
	$(LINKXX) $(LINKFLAGS) $(LINK_CXXFLAGS) -o $@ $< $(LINK_LIB) $(LDLIBS)

$(LIBDIR)/librast3dmhd.a: $(LIB_OBJS) $(GPU_OBJS)
	ar rcs $@ $^

$(OBJDIR)/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

-include $(wildcard $(OBJDIR)/*.d)

$(OBJDIR)/%.o: src/%.cu
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OBJDIR)/main.o: src/main.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# --------------------------------------------------------------------------
# Tests.  Each links the library; MPI-flavoured tests run single-rank unless
# a runner script maps them onto multiple ranks.
# --------------------------------------------------------------------------
$(TEST_BINS): tests/%_test: $(OBJDIR)/%_test.o $(LIBDIR)/librast3dmhd.a
	$(LINKXX) $(LINKFLAGS) $(LINK_CXXFLAGS) $(GTEST_INC) -o $@ $< $(LINK_LIB) \
	  $(if $(filter $@,$(MULTI)),/scratch/zyi/.kez/env/system/lib64/libgtest.a,$(GTEST_LIB)) $(LDLIBS)

# MPI tests that provide their own main() (MPI_Init) link gtest without
# gtest_main, and are run under mpirun by 'make test_multi' rather than the
# single-process 'make test' (they need NPEZ >= 2).
MULTI := tests/communicate_test tests/static_test tests/step_test tests/gpu_equivalence_test
# GPU-only tests (require BACKEND=gpu)
GPU_TESTS := tests/gpu_equivalence_test

$(OBJDIR)/%_test.o: tests/%_test.cpp
	$(CXX) $(CXXFLAGS) $(GTEST_INC) -c -o $@ $<

-include $(wildcard $(OBJDIR)/*.d)

# Single-process tests (run without MPI).  The multi-rank tests are launched
# separately (see the project TODO: cluster runs are user-triggered).
SINGLE := $(filter-out $(MULTI),$(TEST_BINS))
test: $(SINGLE)
	@for t in $(SINGLE); do echo; echo "== $$t =="; ./$$t || exit 1; done

test_multi: $(filter-out $(GPU_TESTS),$(MULTI))
	mpirun -np 4 ./tests/communicate_test
	mpirun -np 4 ./tests/static_test
	mpirun -np 4 ./tests/step_test

test_gpu: $(GPU_TESTS)
	mpirun -np 4 ./tests/gpu_equivalence_test

clean:
	rm -rf $(OBJDIR) $(BINDIR) $(LIBDIR) $(TESTS)

.PHONY: all lib test test_multi clean
