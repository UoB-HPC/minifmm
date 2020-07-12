CC_GNU=g++
CC_INTEL=icpc
CC_CLANG=clang++
CC_ARM=armclang++
CC_CRAY=CC
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

CFLAGS_CLANG=-std=c++11 -Ofast $(ARCH_CFLAGS) -fopenmp
CFLAGS_GNU=-std=c++11 -Ofast -fno-cx-limited-range $(ARCH_CFLAGS) -fopenmp
CFLAGS_INTEL=-std=c++11 -Ofast -x$(ARCH) -qopenmp
CFLAGS_ARM=-std=c++11 -Ofast $(ARCH_CFLAGS) -fopenmp
CFLAGS_CRAY=-std=c++11 -Ofast -fopenmp
CFLAGS=$(CFLAGS_$(COMPILER)) -Wall -g

LIBS=-fopenmp
