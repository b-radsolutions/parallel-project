
CC=mpicxx
CFLAGS=-std=c++11 -Wall -Wextra -I. -O3
ODIR=obj

DEPS = gram-schmidt.hpp matrix-operations.hpp mpi-helper.hpp orthogonality-test.hpp
_OBJS = main.o gram-schmidt.o orthogonality-test.o mpi-helper.o serial-linalg.o
OBJS = $(patsubst %,$(ODIR)/%,$(_OBJS))

OUTPUT=run.out

$(OUTPUT): $(OBJS) cuda-gram-schmidt.o
	$(CC) -g cuda-gram-schmidt.o $(OBJS) \
	-o $(OUTPUT) -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++

$(ODIR):
	mkdir -p $(ODIR)

$(ODIR)/%.o: %.cpp $(DEPS) $(ODIR)
	$(CC) -c -o $@ $< $(CFLAGS)

.PHONY: run-all
run-all: $(OUTPUT)
	mkdir -p out/4
	mkdir -p out/8
	mkdir -p out/16
	mkdir -p out/32
	sbatch -N 1 -p el8-rpi --gres=gpu -t 30 ./slurm.sh
	sbatch -N 2 -p el8-rpi --gres=gpu -t 30 ./slurm.sh
	sbatch -N 4 -p el8-rpi --gres=gpu -t 30 ./slurm.sh
	sbatch -N 8 -p el8-rpi --gres=gpu -t 30 ./slurm.sh

cuda-gram-schmidt.o: cuda-gram-schmidt.cu
	nvcc cuda-gram-schmidt.cu -c -o cuda-gram-schmidt.o

.PHONY: test
test: test.out
	@./test.out

test.out: test.cu cuda-gram-schmidt.cu
	nvcc -g -G -gencode arch=compute_70,code=sm_70 cuda-gram-schmidt.cu test.cu -o test.out

.PHONY: clean 
clean:
	rm -f $(OBJS)
	rm ./$(OUTPUT)
	rm -rf ./*.dSYM

ortho:
	mkdir -p out
	g++ calculate-norms.cpp ./tools/read_matrix_serial.cpp orthogonality-test.cpp serial-linalg.cpp -o ortho
	./ortho
	