
generate.out: generate-matrix.cpp matrix-writer.cpp
	g++ -O3 matrix-writer.cpp generate-matrix.cpp -o generate.out

.PHONY: generate-all
generate-all: generate.out
	mkdir -p gen
	./generate.out gen/16.mtx 16
	./generate.out gen/32.mtx 32
	./generate.out gen/64.mtx 64
	./generate.out gen/128.mtx 128
	./generate.out gen/256.mtx 256
	./generate.out gen/512.mtx 512
	./generate.out gen/1024.mtx 1024
	
.PHONY: clean
clean:
	rm -rf gen
	rm -rf ./*.mtx
