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

LIBS += -lgsl -lgslcblas -lm

ifeq ($(CC), icc)
	CFLAGS += 
else
	CFLAGS += 
endif

HEADERS = $(wildcard *.h)
OBJECTS=main.o util.o tree.o parse_args.o kernels.o traversal.o verify.o initialise.o

default: fmm

$(TARGET): $(OBJECTS)	
	$(CC) $(OBJECTS) -Wall $(LIBS) -o $@

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f *.o
	-rm -f $(TARGET)