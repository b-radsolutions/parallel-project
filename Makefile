
.PHONY: test
test: test.out
	@./test.out

test.out:
	nvcc -g -G -gencode arch=compute_70,code=sm_70 cuda-gram-schmidt.cu test.cu -o test.out

run.out:
	nvcc -g -G cuda-gram-schmidt.cu -c -o cuda-gram-schmidt.o
	g++ -g modified-gram-schmidt.cpp -o modified-gram-schmidt.o
	g++ -g orthogonality-test.cpp -o orthogonality-test.o
	g++ -g test.cpp -o testOrthTest.o
	g++ cuda-gram-schmidt.o modified-gram-schmidt.o orthogonality-test.o -o run.out -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++


