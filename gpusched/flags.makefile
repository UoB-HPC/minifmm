CC_NVCC=nvcc
CC=$(CC_$(COMPILER))

ifneq ($(COMPILER), NVCC)
	$(error Only NVCC support for this version of MiniFMM)
endif

ARCH=sm_60

CFLAGS_NVCC=-DNWORKERS=1 -std=c++11 -O3 -ftz=true --use_fast_math -x cu -Xcompiler -fopenmp -arch=$(ARCH)
CFLAGS=$(CFLAGS_$(COMPILER))

LIBS=-Xcompiler -fopenmp

