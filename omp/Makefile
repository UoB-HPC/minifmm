ifndef COMPILER
define compiler_help
Set COMPILER to change flags (defaulting to GNU).
Available compilers are :
	CLANG CRAY GNU INTEL

endef
$(info $(compiler_help))
COMPILER=GNU
endif

ARCH=broadwell

COMPILER_GNU=gcc
COMPILER_INTEL=icc
COMPILER_CLANG=clang
COMPILER_CRAY=cc
CC=$(COMPILER_$(COMPILER))

FLAGS_GNU = -O3 -std=gnu99 -Wall -march=$(ARCH)
FLAGS_INTEL = -O3 -std=gnu99 -Wall -x$(ARCH)
FLAGS_CRAY = -O3
FLAGS_CLANG = -O3 -std=gnu99
CFLAGS = $(FLAGS_$(COMPILER))

LINKER_OPTS_GNU = -Wall
LINKER_OPTS_INTEL = -Wall
LINKER_OPTS_CRAY =
LINKER_OPTS_CLANG = -Wall
LINKER_OPTS = $(LINKER_OPTS_$(COMPILER))

OMP_GNU   = -fopenmp
OMP_INTEL = -qopenmp
OMP_CRAY  = -homp
OMP_CLANG = -fopenmp=libomp
CFLAGS += $(OMP_$(COMPILER))
LIBS = $(OMP_$(COMPILER))

TARGET=fmm
TYPE=DOUBLE

ifeq ($(TYPE), DOUBLE)
	CFLAGS += -DUSE_DOUBLE
else
	CFLAGS += -DUSE_FLOAT
endif


HOSTNAME=$(shell hostname)
ifeq ($(HOSTNAME), swan)
	LIBS += -L /lus/scratch/p02340/gsl/lib
	CFLAGS += -I /lus/scratch/p02340/gsl/include
endif

LIBS += -lgsl -lgslcblas -lm

REF_DIR=$(abspath $(shell pwd)/../ref)
CFLAGS += -I $(REF_DIR)

HEADERS=$(wildcard $(REF_DIR)/*.h)
REF_SRCS=$(wildcard $(REF_DIR)/*.c)
DIFF_SRCS=$(wildcard *.c)

SRCS = $(addprefix $(REF_DIR)/, $(filter-out $(DIFF_SRCS), $(notdir $(REF_SRCS))))
SRCS += $(DIFF_SRCS)
OBJS = $(patsubst %.c,%.o,$(SRCS))

default: fmm

print-%  : ; @echo $* = $($*)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) $(LINKER_OPTS) $(LIBS) -o $@

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f $(OBJS)
	-rm -f $(TARGET)