
run.out:
	nvcc -g -G cuda-gram-schmidt.cu -c -o cuda-gram-schmidt.o
	g++ -g modified-gram-schmidt.cpp -o modified-gram-schmidt.o
	g++ cuda-gram-schmidt.o modified-gram-schmidt.o -o run.out -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++

.PHONY: test
test: test.out
	@./test.out

test.out: test.cu cuda-gram-schmidt.cu
	nvcc -g -G -gencode arch=compute_70,code=sm_70 cuda-gram-schmidt.cu test.cu -o test.out

scream: scream_matrix.cpp ./tools/read_matrix_serial.cpp
	g++ -O3 scream_matrix.cpp ./tools/read_matrix_serial.cpp -o scream
