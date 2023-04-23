
CC=mpicxx

run.out:
	nvcc -g -G cuda-gram-schmidt.cu -c -o cuda-gram-schmidt.o
	$(CC) -g -c modified-gram-schmidt.cpp -o modified-gram-schmidt.o
	$(CC) -g -c gram-schmidt.cpp -o gram-schmidt.o
	$(CC) -g -c orthogonality-test.cpp -o orthogonality-test.o 
	$(CC) -g -c mpi-helper.cpp -o mpi-helper.o
	$(CC) -g -c serial-linalg.cpp -o serial-linalg.o
	$(CC) -std=c++11 -g -c main.cpp -o main.o
	$(CC) cuda-gram-schmidt.o modified-gram-schmidt.o orthogonality-test.o main.o gram-schmidt.o mpi-helper.o serial-linalg.o \
	-o run.out -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++

.PHONY: test
test: test.out
	@./test.out

test.out: test.cu cuda-gram-schmidt.cu
	nvcc -g -G -gencode arch=compute_70,code=sm_70 cuda-gram-schmidt.cu test.cu -o test.out

scream: scream_matrix.cpp ./tools/read_matrix_serial.cpp
	g++ -O3 scream_matrix.cpp ./tools/read_matrix_serial.cpp -o scream
