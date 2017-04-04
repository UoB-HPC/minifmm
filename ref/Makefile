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

LIBS=
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

HEADERS = $(wildcard *.h)
OBJECTS=main.o util.o tree.o parse_args.o kernels.o traversal.o verify.o initialise.o

default: fmm

$(TARGET): $(OBJECTS)	
	$(CC) $(OBJECTS) -Wall $(LIBS) -o $@

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	-rm -f *.o
	-rm -f $(TARGET)