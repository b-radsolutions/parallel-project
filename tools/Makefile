
generate.out: generate-matrix.cpp matrix-writer.cpp
	g++ -O3 matrix-writer.cpp generate-matrix.cpp -o generate.out

res.out: resurrect.cpp
	g++ -O3 resurrect.cpp -o res.out

res: res.out
	./res.out ../../1node/out/ClassicParallel64by64densepart 4 ./tmp

.PHONY: generate-all
generate-all: generate.out
	mkdir -p ../data
	mkdir -p ../data/dense
	./generate.out ../data/dense/4.mtx 4 1
	./generate.out ../data/dense/16.mtx 16 1
	./generate.out ../data/dense/32.mtx 32 1
	./generate.out ../data/dense/64.mtx 64 1
	./generate.out ../data/dense/128.mtx 128 1
	./generate.out ../data/dense/256.mtx 256 1
	./generate.out ../data/dense/512.mtx 512 1
	./generate.out ../data/dense/1024.mtx 1024 1
	./generate.out ../data/dense/2048.mtx 2048 1
	./generate.out ../data/dense/4096.mtx 4096 1
	./generate.out ../data/dense/8192.mtx 8192 1

	mkdir -p ../data/sparse
	./generate.out ../data/sparse/4.mtx 4 2
	./generate.out ../data/sparse/16.mtx 16 2
	./generate.out ../data/sparse/32.mtx 32 2
	./generate.out ../data/sparse/64.mtx 64 2
	./generate.out ../data/sparse/128.mtx 128 2
	./generate.out ../data/sparse/256.mtx 256 2
	./generate.out ../data/sparse/512.mtx 512 2
	./generate.out ../data/sparse/1024.mtx 1024 2
	./generate.out ../data/sparse/2048.mtx 2048 2
	./generate.out ../data/sparse/4096.mtx 4096 2
	./generate.out ../data/sparse/8192.mtx 8192 2

	mkdir -p ../data/well-conditioned
	./generate.out ../data/well-conditioned/4.mtx 4 3
	./generate.out ../data/well-conditioned/16.mtx 16 3
	./generate.out ../data/well-conditioned/32.mtx 32 3
	./generate.out ../data/well-conditioned/64.mtx 64 3
	./generate.out ../data/well-conditioned/128.mtx 128 3
	./generate.out ../data/well-conditioned/256.mtx 256 3
	./generate.out ../data/well-conditioned/512.mtx 512 3
	./generate.out ../data/well-conditioned/1024.mtx 1024 3
	./generate.out ../data/well-conditioned/2048.mtx 2048 3
	./generate.out ../data/well-conditioned/4096.mtx 4096 3
	./generate.out ../data/well-conditioned/8192.mtx 8192 3

	mkdir -p ../data/ill-conditioned
	./generate.out ../data/ill-conditioned/4.mtx 4 4
	./generate.out ../data/ill-conditioned/16.mtx 16 4
	./generate.out ../data/ill-conditioned/32.mtx 32 4
	./generate.out ../data/ill-conditioned/64.mtx 64 4
	./generate.out ../data/ill-conditioned/128.mtx 128 4
	./generate.out ../data/ill-conditioned/256.mtx 256 4
	./generate.out ../data/ill-conditioned/512.mtx 512 4
	./generate.out ../data/ill-conditioned/1024.mtx 1024 4
	./generate.out ../data/ill-conditioned/2048.mtx 2048 4
	./generate.out ../data/ill-conditioned/4096.mtx 4096 4
	./generate.out ../data/ill-conditioned/8192.mtx 8192 4
	
.PHONY: clean
clean:
	rm -rf gen
	rm -rf ./*.mtx
