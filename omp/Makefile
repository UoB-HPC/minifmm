CC=gcc
CFLAGS=-g -Wall -O3 -std=gnu99
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
else
	CFLAGS += -march=native
endif

LIBS += -lgsl -lgslcblas -lm -fopenmp

ifeq ($(CC), icc)
	CFLAGS += -qopenmp
else
	CFLAGS += -fopenmp
endif

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
	$(CC) $(OBJS) -Wall $(LIBS) -o $@

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f $(OBJS)
	-rm -f $(TARGET)