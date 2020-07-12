COMPILER ?= GNU
ARCH ?= native
MODEL = omp-task

EXES=$(addprefix fmm., $(MODEL))
SINGLE=$(addsuffix .single, $(EXES))
DOUBLE=$(addsuffix .double, $(EXES))

default: fmm.$(MODEL)

include $(MODEL)/flags.makefile

.PHONY: default clean

COMMON_HEADERS=$(wildcard common/*.hh)
COMMON_INC=-I./common

main.o: main.cc $(MODEL)/*.hh $(COMMON_HEADERS)
	$(CC) $(CFLAGS) $(EXTRA_FLAGS) -I./$(MODEL) $(COMMON_INC) main.cc -c

fmm.$(MODEL): main.o
	$(CC) main.o -o $@ $(LIBS) 

#fmm.%.single: main.cc %/*.hh $(COMMON_HEADERS)
#	$(CC) $(CFLAGS) -I $* $(COMMON_INC) main.cc -o $@ $(LIBS)
#
#fmm.%.double: main.cc %/*.hh $(COMMON_HEADERS)
#	$(CC) $(CFLAGS) -DFMM_DOUBLE -I $* $(COMMON_INC) main.cc -o $@ $(LIBS)

clean:
	-rm -f fmm.* main.o

