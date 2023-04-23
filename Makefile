
CC=mpicxx
CFLAGS=-std=c++11 -Wall -Wextra -I. -O3
ODIR=obj

DEPS = gram-schmidt.hpp matrix-operations.hpp mpi-helper.hpp orthogonality-test.hpp
_OBJS = main.o modified-gram-schmidt.o gram-schmidt.o orthogonality-test.o mpi-helper.o serial-linalg.o
OBJS = $(patsubst %,$(ODIR)/%,$(_OBJS))

OUTPUT=run.out

$(OUTPUT): $(OBJS) cuda-gram-schmidt.o
	$(CC) cuda-gram-schmidt.o $(OBJS) \
	-o $(OUTPUT) -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++

$(ODIR):
	mkdir -p $(ODIR)

$(ODIR)/%.o: %.cpp $(DEPS) $(ODIR)
	$(CC) -g -c -o $@ $< $(CFLAGS)

cuda-gram-schmidt.o: cuda-gram-schmidt.cu
	nvcc -g -G cuda-gram-schmidt.cu -c -o cuda-gram-schmidt.o

.PHONY: test
test: test.out
	@./test.out

test.out: test.cu cuda-gram-schmidt.cu
	nvcc -g -G -gencode arch=compute_70,code=sm_70 cuda-gram-schmidt.cu test.cu -o test.out

scream: scream_matrix.cpp ./tools/read_matrix_serial.cpp
	g++ -O3 scream_matrix.cpp ./tools/read_matrix_serial.cpp -o scream

.PHONY: clean 
clean:
	rm -f $(OBJS)
	rm ./$(OUTPUT)
	rm -rf ./*.dSYM
