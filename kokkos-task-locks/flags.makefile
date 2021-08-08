include $(KOKKOS_PATH)/Makefile.kokkos

CC_GNU=g++
CC_INTEL=icpc
CC_CLANG=clang++
CC_ARM=armclang++
CC_CRAY=CC
CC_NVCC=nvcc_wrapper
CC=$(CC_$(COMPILER))

UNAME=$(shell uname -m)
ifeq ($(UNAME), aarch64)
  ARCH_CFLAGS = -mcpu=$(ARCH) -mtune=$(ARCH) 
  ifeq ($(COMPILER), GNU)
    ARCH_CFLAGS += -mlow-precision-recip-sqrt
  endif
endif
ifeq ($(UNAME), x86_64)
  ARCH_CFLAGS = -march=$(ARCH)
endif

CFLAGS_CLANG=-Ofast $(ARCH_CFLAGS) -fopenmp
CFLAGS_GNU=-Ofast -fno-cx-limited-range $(ARCH_CFLAGS) -fopenmp
CFLAGS_INTEL=-Ofast -x$(ARCH_CFLAGS) -qopenmp
CFLAGS_ARM=-Ofast $(ARCH_CFLAGS) -fopenmp
CFLAGS_CRAY=-fopenmp
CFLAGS_NVCC=-O3 -ftz=true --use_fast_math 
CFLAGS=$(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CFLAGS_$(COMPILER)) 

LIBS=$(KOKKOS_LDFLAGS) $(KOKKOS_LIBS)

