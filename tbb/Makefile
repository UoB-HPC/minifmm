ifndef COMPILER
define compiler_help
Set COMPILER to change flags (defaulting to GNU).
Available compilers are :
	CLANG CRAY GNU INTEL

endef
$(info $(compiler_help))
COMPILER=GNU
endif

ARCH=native

ifndef TBB_ROOT
    $(error TBB_ROOT not set)
endif

COMPILER_GNU=g++
COMPILER_INTEL=icpc
COMPILER_CLANG=clang++
COMPILER_CRAY=CC
CC=$(COMPILER_$(COMPILER))

FLAGS_GNU = -O3 -std=gnu++11 -Wall -march=$(ARCH)
FLAGS_INTEL = -O3 -std=gnu++11 -Wall -x$(ARCH)
FLAGS_CRAY = -O3
FLAGS_CLANG = -O3 -std=c++11 -march=$(ARCH)
CFLAGS = $(FLAGS_$(COMPILER))

LINKER_OPTS_GNU = -Wall
LINKER_OPTS_INTEL = -Wall
LINKER_OPTS_CRAY =
LINKER_OPTS_CLANG = -Wall
LINKER_OPTS = $(LINKER_OPTS_$(COMPILER))

CFLAGS+=-I $(TBB_ROOT)/include
TBB_LIBS_GNU=-L $(TBB_ROOT)/lib/intel64/gcc4.7
LIBS+=$(TBB_LIBS_$(COMPILER))

LIBS+=-ltbb

TARGET=fmm
TYPE=DOUBLE

ifeq ($(TYPE), DOUBLE)
	CFLAGS += -DUSE_DOUBLE
else
	CFLAGS += -DUSE_FLOAT
endif

LIBS += -lm

REF_DIR=$(abspath $(shell pwd)/../ref)
CFLAGS += -I $(REF_DIR)

HEADERS=$(wildcard $(REF_DIR)/*.h)
REF_SRCS=$(wildcard $(REF_DIR)/*.c)
DIFF_SRCS=$(wildcard *.c)

SRCS = $(addprefix $(REF_DIR)/, $(filter-out $(DIFF_SRCS), $(notdir $(REF_SRCS))))
SRCS += $(DIFF_SRCS)
OBJS = $(patsubst %.c,%.o,$(SRCS))

default: $(TARGET)

print-%  : ; @echo $* = $($*)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) $(LINKER_OPTS) $(LIBS) -o $@

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f $(OBJS)
	-rm -f $(TARGET)
