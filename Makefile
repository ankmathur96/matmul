
# We will compile your code with the PrgEnv-intel and PrgEnv-gnu
# programming environments and use the best score.  The PrgEnv-intel
# environment is the default on Cori.  To switch to another environment, use
# something like `module swap PrgEnv-intel PrgEnv-gnu`.
#
# On Cori, we will benchmark your DGEMM's performance against the performance
# of the default vendor-tuned DGEMM. This is done in benchmark-blas.
#
# NERSC's cc and CC compiler wrappers link benchmark-blas's call to dgemm
# to the correct implementation automatically. If you wish to compare with
# other BLAS implementations, check the NERSC documentation.

CC = cc 
OPT = -O2
CFLAGS = -Wall -std=gnu99 $(OPT) -fopenmp -lblas -mavx2 -mavx -mfma
LDFLAGS = -Wall
# librt is needed for clock_gettime
LDLIBS = -lrt -fopenmp -lblas

targets = benchmark-naive benchmark-blocked benchmark-blas benchmark-blocked2-2
objects = benchmark.o dgemm-naive.o dgemm-blocked.o dgemm-blas.o dgemm-blocked2-2.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark.o dgemm-naive.o 
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked : benchmark.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark.o dgemm-blas.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked2-2: benchmark.o dgemm-blocked2-2.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)
