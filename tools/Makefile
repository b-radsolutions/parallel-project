
generate.out: generate-matrix.cpp matrix-writer.cpp
	g++ -O3 matrix-writer.cpp generate-matrix.cpp -o generate.out

.PHONY: generate-all
generate-all: generate.out
	mkdir -p dense
	./generate.out dense/16.mtx 16 1
	./generate.out dense/32.mtx 32 1
	./generate.out dense/64.mtx 64 1
	./generate.out dense/128.mtx 128 1
	./generate.out dense/256.mtx 256 1
	./generate.out dense/512.mtx 512 1
	./generate.out dense/1024.mtx 1024 1

	mkdir -p sparse
	./generate.out sparse/16.mtx 16 2
	./generate.out sparse/32.mtx 32 2
	./generate.out sparse/64.mtx 64 2
	./generate.out sparse/128.mtx 128 2
	./generate.out sparse/256.mtx 256 2
	./generate.out sparse/512.mtx 512 2
	./generate.out sparse/1024.mtx 1024 2

	mkdir -p well-conditioned
	./generate.out well-conditioned/16.mtx 16 3
	./generate.out well-conditioned/32.mtx 32 3
	./generate.out well-conditioned/64.mtx 64 3
	./generate.out well-conditioned/128.mtx 128 3
	./generate.out well-conditioned/256.mtx 256 3
	./generate.out well-conditioned/512.mtx 512 3
	./generate.out well-conditioned/1024.mtx 1024 3

	mkdir -p ill-conditioned
	./generate.out ill-conditioned/16.mtx 16 4
	./generate.out ill-conditioned/32.mtx 32 4
	./generate.out ill-conditioned/64.mtx 64 4
	./generate.out ill-conditioned/128.mtx 128 4
	./generate.out ill-conditioned/256.mtx 256 4
	./generate.out ill-conditioned/512.mtx 512 4
	./generate.out ill-conditioned/1024.mtx 1024 4
	
.PHONY: clean
clean:
	rm -rf gen
	rm -rf ./*.mtx
