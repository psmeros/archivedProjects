SRC=src
all: conjugategradient_seq conjugategradient_mpi conjugategradient_cuda

OBJS_SEQ=${SRC}/mmio.o ${SRC}/util.o ${SRC}/second.o ${SRC}/cg_seq.o ${SRC}/cg_blas_seq.o
conjugategradient_seq: CFLAGS+=-Wall -pedantic -O3 -fPIC -Wno-unused-variable -Wno-implicit-function-declaration -Wno-unused-result -I$(OPENBLAS_ROOT)/include
conjugategradient_seq: LDFLAGS+=-lm -L$(OPENBLAS_ROOT)/lib -lopenblas
conjugategradient_seq: CC=gcc
conjugategradient_seq: $(OBJS_SEQ)
	$(CC) $(OBJS_SEQ) $(LDFLAGS) -o $@

OBJS_MPI=${SRC}/mmio.o ${SRC}/util.o ${SRC}/second.o ${SRC}/cg_mpi.o ${SRC}/cg_blas_mpi.o
conjugategradient_mpi: CFLAGS+=-Wall -pedantic -O3 -fPIC -Wno-unused-variable -Wno-implicit-function-declaration -Wno-unused-result -I$(OPENBLAS_ROOT)/include
conjugategradient_mpi: LDFLAGS+=-lm -fopenmp -L$(OPENBLAS_ROOT)/lib -lopenblas
conjugategradient_mpi: CC=mpicc
conjugategradient_mpi: $(OBJS_MPI)
	$(CC) $(OBJS_MPI) $(LDFLAGS) -o $@

${SRC}/cg_cuda.o: ${SRC}/cg_cuda.cu
	nvcc -O3 -I$(OPENBLAS_ROOT)/include -lm  -L$(OPENBLAS_ROOT)/lib -lopenblas -c ${SRC}/cg_cuda.cu -o ${SRC}/cg_cuda.o

OBJS_CUDA=${SRC}/mmio.o ${SRC}/util.o ${SRC}/second.o ${SRC}/cg_blas_cuda.o ${SRC}/cg_cuda.o
conjugategradient_cuda: CFLAGS+=-O3 -I$(OPENBLAS_ROOT)/include
conjugategradient_cuda: LDFLAGS+=-lm -L$(OPENBLAS_ROOT)/lib -lopenblas
conjugategradient_cuda: CC=nvcc
conjugategradient_cuda: $(OBJS_CUDA)
	$(CC) $(OBJS_CUDA) $(LDFLAGS) -o $@


run_strong_scaling:
	./conjugategradient_seq -M data/10000.mtx
	mpiexec -np 1 ./conjugategradient_mpi -M data/10000.mtx
	mpiexec -np 10 ./conjugategradient_mpi -M data/10000.mtx
	mpiexec -np 20 ./conjugategradient_mpi -M data/10000.mtx
	mpiexec -np 50 ./conjugategradient_mpi -M data/10000.mtx
	./conjugategradient_cuda -M data/10000.mtx -np 50
	./conjugategradient_cuda -M data/10000.mtx -np 100
	./conjugategradient_cuda -M data/10000.mtx -np 1000
	./conjugategradient_cuda -M data/10000.mtx -np 10000

run_weak_scaling:
	./conjugategradient_seq -M data/20.mtx
	./conjugategradient_seq -M data/66.mtx
	./conjugategradient_seq -M data/100.mtx
	./conjugategradient_seq -M data/1050.mtx
	./conjugategradient_seq -M data/4098.mtx
	./conjugategradient_seq -M data/10000.mtx
	mpiexec -np 20 ./conjugategradient_mpi -M data/20.mtx
	mpiexec -np 66 ./conjugategradient_mpi -M data/66.mtx
	mpiexec -np 100 ./conjugategradient_mpi -M data/100.mtx
	./conjugategradient_cuda -M data/20.mtx -np 20
	./conjugategradient_cuda -M data/66.mtx -np 66
	./conjugategradient_cuda -M data/100.mtx -np 100
	./conjugategradient_cuda -M data/1050.mtx -np 1050
	./conjugategradient_cuda -M data/4098.mtx -np 4098
	./conjugategradient_cuda -M data/10000.mtx -np 10000


run-seq:
	./conjugategradient_seq -M data/10000.mtx

run-mpi:
	mpiexec -np 64 ./conjugategradient_mpi -M data/10000.mtx

run-cuda:
	./conjugategradient_cuda -M data/10000.mtx -np 10000


clean:
	rm -Rf conjugategradient* ${SRC}/*.o ${SRC}/*~
