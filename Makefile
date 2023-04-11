
.PHONY: test
test: test.out
	@./test.out

test.out:
	nvcc -g -G -gencode arch=compute_70,code=sm_70 cuda-gram-schmidt.cu test.cu -o test.out
