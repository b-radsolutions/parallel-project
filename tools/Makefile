
generate.out: generate-matrix.cpp matrix-writer.cpp
	g++ -O3 matrix-writer.cpp generate-matrix.cpp -o generate.out

res: resurrect.cpp
	g++ -O3 resurrect.cpp -o res

scream: scream_matrix.cpp read_matrix_serial.cpp
	g++ -O3 scream_matrix.cpp read_matrix_serial.cpp -o scream

ortho:
	g++ calculate-norms.cpp read_matrix_serial.cpp orthogonality-test.cpp -o ortho

INPUT_LOC=../../results/out
OUTPUT_LOC=../../results/reconstruct
.PHONY: process
process: res ortho
	@echo "Reconstruct matrices from partials"
	@./res $(INPUT_LOC)/4/ModifiedParallel4by4dense_part_ 4 $(OUTPUT_LOC)/4/MP4dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel16by16dense_part_ 4 $(OUTPUT_LOC)/4/MP16dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel32by32dense_part_ 4 $(OUTPUT_LOC)/4/MP32dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel64by64dense_part_ 4 $(OUTPUT_LOC)/4/MP64dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel128by128dense_part_ 4 $(OUTPUT_LOC)/4/MP128dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel256by256dense_part_ 4 $(OUTPUT_LOC)/4/MP256dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel512by512dense_part_ 4 $(OUTPUT_LOC)/4/MP512dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel1024by1024dense_part_ 4 $(OUTPUT_LOC)/4/MP1024dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel2048by2048dense_part_ 4 $(OUTPUT_LOC)/4/MP2048dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel4096by4096dense_part_ 4 $(OUTPUT_LOC)/4/MP4096dense.mtx > /dev/null
	
	@./res $(INPUT_LOC)/8/ModifiedParallel4by4dense_part_ 8 $(OUTPUT_LOC)/8/MP4dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel16by16dense_part_ 8 $(OUTPUT_LOC)/8/MP16dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel32by32dense_part_ 8 $(OUTPUT_LOC)/8/MP32dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel64by64dense_part_ 8 $(OUTPUT_LOC)/8/MP64dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel128by128dense_part_ 8 $(OUTPUT_LOC)/8/MP128dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel256by256dense_part_ 8 $(OUTPUT_LOC)/8/MP256dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel512by512dense_part_ 8 $(OUTPUT_LOC)/8/MP512dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel1024by1024dense_part_ 8 $(OUTPUT_LOC)/8/MP1024dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel2048by2048dense_part_ 8 $(OUTPUT_LOC)/8/MP2048dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel4096by4096dense_part_ 8 $(OUTPUT_LOC)/8/MP4096dense.mtx > /dev/null

	@./res $(INPUT_LOC)/16/ModifiedParallel4by4dense_part_ 16 $(OUTPUT_LOC)/16/MP4dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel16by16dense_part_ 16 $(OUTPUT_LOC)/16/MP16dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel32by32dense_part_ 16 $(OUTPUT_LOC)/16/MP32dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel64by64dense_part_ 16 $(OUTPUT_LOC)/16/MP64dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel128by128dense_part_ 16 $(OUTPUT_LOC)/16/MP128dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel256by256dense_part_ 16 $(OUTPUT_LOC)/16/MP256dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel512by512dense_part_ 16 $(OUTPUT_LOC)/16/MP512dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel1024by1024dense_part_ 16 $(OUTPUT_LOC)/16/MP1024dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel2048by2048dense_part_ 16 $(OUTPUT_LOC)/16/MP2048dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel4096by4096dense_part_ 16 $(OUTPUT_LOC)/16/MP4096dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel8192by8192dense_part_ 16 $(OUTPUT_LOC)/16/MP8192dense.mtx > /dev/null

	@./res $(INPUT_LOC)/32/ModifiedParallel4by4dense_part_ 32 $(OUTPUT_LOC)/32/MP4dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel16by16dense_part_ 32 $(OUTPUT_LOC)/32/MP16dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel32by32dense_part_ 32 $(OUTPUT_LOC)/32/MP32dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel64by64dense_part_ 32 $(OUTPUT_LOC)/32/MP64dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel128by128dense_part_ 32 $(OUTPUT_LOC)/32/MP128dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel256by256dense_part_ 32 $(OUTPUT_LOC)/32/MP256dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel512by512dense_part_ 32 $(OUTPUT_LOC)/32/MP512dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel1024by1024dense_part_ 32 $(OUTPUT_LOC)/32/MP1024dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel2048by2048dense_part_ 32 $(OUTPUT_LOC)/32/MP2048dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel4096by4096dense_part_ 32 $(OUTPUT_LOC)/32/MP4096dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel8192by8192dense_part_ 32 $(OUTPUT_LOC)/32/MP8192dense.mtx > /dev/null

	@echo "Calculating norms"
	./ortho -r $(OUTPUT_LOC)/4/
	./ortho -r $(OUTPUT_LOC)/8/
	./ortho -r $(OUTPUT_LOC)/16/
	./ortho -r $(OUTPUT_LOC)/32/


CLASSIC_OUTPUT_LOC=../../results/reconstruct
.PHONY: process-other
process-other: res ortho
	@echo "Reconstruct matrices from partials"
	@echo "Sparse"
	@./res $(INPUT_LOC)/4/ModifiedParallel4by4sparse_part_ 4 $(OUTPUT_LOC)/4sp/MP4dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel16by16sparse_part_ 4 $(OUTPUT_LOC)/4sp/MP16dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel32by32sparse_part_ 4 $(OUTPUT_LOC)/4sp/MP32dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel64by64sparse_part_ 4 $(OUTPUT_LOC)/4sp/MP64dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel128by128sparse_part_ 4 $(OUTPUT_LOC)/4sp/MP128dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel256by256sparse_part_ 4 $(OUTPUT_LOC)/4sp/MP256dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel512by512sparse_part_ 4 $(OUTPUT_LOC)/4sp/MP512dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel1024by1024sparse_part_ 4 $(OUTPUT_LOC)/4sp/MP1024dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel2048by2048sparse_part_ 4 $(OUTPUT_LOC)/4sp/MP2048dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel4096by4096sparse_part_ 4 $(OUTPUT_LOC)/4sp/MP4096dense.mtx > /dev/null
	
	@./res $(INPUT_LOC)/8/ModifiedParallel4by4sparse_part_ 8 $(OUTPUT_LOC)/8sp/MP4dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel16by16sparse_part_ 8 $(OUTPUT_LOC)/8sp/MP16dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel32by32sparse_part_ 8 $(OUTPUT_LOC)/8sp/MP32dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel64by64sparse_part_ 8 $(OUTPUT_LOC)/8sp/MP64dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel128by128sparse_part_ 8 $(OUTPUT_LOC)/8sp/MP128dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel256by256sparse_part_ 8 $(OUTPUT_LOC)/8sp/MP256dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel512by512sparse_part_ 8 $(OUTPUT_LOC)/8sp/MP512dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel1024by1024sparse_part_ 8 $(OUTPUT_LOC)/8sp/MP1024dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel2048by2048sparse_part_ 8 $(OUTPUT_LOC)/8sp/MP2048dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel4096by4096sparse_part_ 8 $(OUTPUT_LOC)/8sp/MP4096dense.mtx > /dev/null

	@./res $(INPUT_LOC)/16/ModifiedParallel4by4sparse_part_ 16 $(OUTPUT_LOC)/16sp/MP4dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel16by16sparse_part_ 16 $(OUTPUT_LOC)/16sp/MP16dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel32by32sparse_part_ 16 $(OUTPUT_LOC)/16sp/MP32dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel64by64sparse_part_ 16 $(OUTPUT_LOC)/16sp/MP64dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel128by128sparse_part_ 16 $(OUTPUT_LOC)/16sp/MP128dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel256by256sparse_part_ 16 $(OUTPUT_LOC)/16sp/MP256dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel512by512sparse_part_ 16 $(OUTPUT_LOC)/16sp/MP512dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel1024by1024sparse_part_ 16 $(OUTPUT_LOC)/16sp/MP1024dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel2048by2048sparse_part_ 16 $(OUTPUT_LOC)/16sp/MP2048dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel4096by4096sparse_part_ 16 $(OUTPUT_LOC)/16sp/MP4096dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel8192by8192sparse_part_ 16 $(OUTPUT_LOC)/16sp/MP8192dense.mtx > /dev/null

	@./res $(INPUT_LOC)/32/ModifiedParallel4by4sparse_part_ 32 $(OUTPUT_LOC)/32sp/MP4dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel16by16sparse_part_ 32 $(OUTPUT_LOC)/32sp/MP16dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel32by32sparse_part_ 32 $(OUTPUT_LOC)/32sp/MP32dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel64by64sparse_part_ 32 $(OUTPUT_LOC)/32sp/MP64dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel128by128sparse_part_ 32 $(OUTPUT_LOC)/32sp/MP128dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel256by256sparse_part_ 32 $(OUTPUT_LOC)/32sp/MP256dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel512by512sparse_part_ 32 $(OUTPUT_LOC)/32sp/MP512dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel1024by1024sparse_part_ 32 $(OUTPUT_LOC)/32sp/MP1024dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel2048by2048sparse_part_ 32 $(OUTPUT_LOC)/32sp/MP2048dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel4096by4096sparse_part_ 32 $(OUTPUT_LOC)/32sp/MP4096dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel8192by8192sparse_part_ 32 $(OUTPUT_LOC)/32sp/MP8192dense.mtx > /dev/null

	@echo "Calculating norms for sparse"
	./ortho -r $(OUTPUT_LOC)/4sp/
	./ortho -r $(OUTPUT_LOC)/8sp/
	./ortho -r $(OUTPUT_LOC)/16sp/
	./ortho -r $(OUTPUT_LOC)/32sp/

	@echo "Ill-conditioned"
	@./res $(INPUT_LOC)/4/ModifiedParallel4by4dense_part_ 4 $(OUTPUT_LOC)/4/MP4dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel16by16dense_part_ 4 $(OUTPUT_LOC)/4/MP16dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel32by32dense_part_ 4 $(OUTPUT_LOC)/4/MP32dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel64by64dense_part_ 4 $(OUTPUT_LOC)/4/MP64dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel128by128dense_part_ 4 $(OUTPUT_LOC)/4/MP128dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel256by256dense_part_ 4 $(OUTPUT_LOC)/4/MP256dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel512by512dense_part_ 4 $(OUTPUT_LOC)/4/MP512dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel1024by1024dense_part_ 4 $(OUTPUT_LOC)/4/MP1024dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel2048by2048dense_part_ 4 $(OUTPUT_LOC)/4/MP2048dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel4096by4096dense_part_ 4 $(OUTPUT_LOC)/4/MP4096dense.mtx > /dev/null
	
	@./res $(INPUT_LOC)/8/ModifiedParallel4by4dense_part_ 8 $(OUTPUT_LOC)/8/MP4dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel16by16dense_part_ 8 $(OUTPUT_LOC)/8/MP16dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel32by32dense_part_ 8 $(OUTPUT_LOC)/8/MP32dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel64by64dense_part_ 8 $(OUTPUT_LOC)/8/MP64dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel128by128dense_part_ 8 $(OUTPUT_LOC)/8/MP128dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel256by256dense_part_ 8 $(OUTPUT_LOC)/8/MP256dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel512by512dense_part_ 8 $(OUTPUT_LOC)/8/MP512dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel1024by1024dense_part_ 8 $(OUTPUT_LOC)/8/MP1024dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel2048by2048dense_part_ 8 $(OUTPUT_LOC)/8/MP2048dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel4096by4096dense_part_ 8 $(OUTPUT_LOC)/8/MP4096dense.mtx > /dev/null

	@./res $(INPUT_LOC)/16/ModifiedParallel4by4dense_part_ 16 $(OUTPUT_LOC)/16/MP4dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel16by16dense_part_ 16 $(OUTPUT_LOC)/16/MP16dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel32by32dense_part_ 16 $(OUTPUT_LOC)/16/MP32dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel64by64dense_part_ 16 $(OUTPUT_LOC)/16/MP64dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel128by128dense_part_ 16 $(OUTPUT_LOC)/16/MP128dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel256by256dense_part_ 16 $(OUTPUT_LOC)/16/MP256dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel512by512dense_part_ 16 $(OUTPUT_LOC)/16/MP512dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel1024by1024dense_part_ 16 $(OUTPUT_LOC)/16/MP1024dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel2048by2048dense_part_ 16 $(OUTPUT_LOC)/16/MP2048dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel4096by4096dense_part_ 16 $(OUTPUT_LOC)/16/MP4096dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel8192by8192dense_part_ 16 $(OUTPUT_LOC)/16/MP8192dense.mtx > /dev/null

	@./res $(INPUT_LOC)/32/ModifiedParallel4by4dense_part_ 32 $(OUTPUT_LOC)/32/MP4dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel16by16dense_part_ 32 $(OUTPUT_LOC)/32/MP16dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel32by32dense_part_ 32 $(OUTPUT_LOC)/32/MP32dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel64by64dense_part_ 32 $(OUTPUT_LOC)/32/MP64dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel128by128dense_part_ 32 $(OUTPUT_LOC)/32/MP128dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel256by256dense_part_ 32 $(OUTPUT_LOC)/32/MP256dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel512by512dense_part_ 32 $(OUTPUT_LOC)/32/MP512dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel1024by1024dense_part_ 32 $(OUTPUT_LOC)/32/MP1024dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel2048by2048dense_part_ 32 $(OUTPUT_LOC)/32/MP2048dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel4096by4096dense_part_ 32 $(OUTPUT_LOC)/32/MP4096dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel8192by8192dense_part_ 32 $(OUTPUT_LOC)/32/MP8192dense.mtx > /dev/null

	@echo "Good"
	@./res $(INPUT_LOC)/4/ModifiedParallel4by4dense_part_ 4 $(OUTPUT_LOC)/4/MP4dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel16by16dense_part_ 4 $(OUTPUT_LOC)/4/MP16dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel32by32dense_part_ 4 $(OUTPUT_LOC)/4/MP32dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel64by64dense_part_ 4 $(OUTPUT_LOC)/4/MP64dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel128by128dense_part_ 4 $(OUTPUT_LOC)/4/MP128dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel256by256dense_part_ 4 $(OUTPUT_LOC)/4/MP256dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel512by512dense_part_ 4 $(OUTPUT_LOC)/4/MP512dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel1024by1024dense_part_ 4 $(OUTPUT_LOC)/4/MP1024dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel2048by2048dense_part_ 4 $(OUTPUT_LOC)/4/MP2048dense.mtx > /dev/null
	@./res $(INPUT_LOC)/4/ModifiedParallel4096by4096dense_part_ 4 $(OUTPUT_LOC)/4/MP4096dense.mtx > /dev/null
	
	@./res $(INPUT_LOC)/8/ModifiedParallel4by4dense_part_ 8 $(OUTPUT_LOC)/8/MP4dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel16by16dense_part_ 8 $(OUTPUT_LOC)/8/MP16dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel32by32dense_part_ 8 $(OUTPUT_LOC)/8/MP32dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel64by64dense_part_ 8 $(OUTPUT_LOC)/8/MP64dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel128by128dense_part_ 8 $(OUTPUT_LOC)/8/MP128dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel256by256dense_part_ 8 $(OUTPUT_LOC)/8/MP256dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel512by512dense_part_ 8 $(OUTPUT_LOC)/8/MP512dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel1024by1024dense_part_ 8 $(OUTPUT_LOC)/8/MP1024dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel2048by2048dense_part_ 8 $(OUTPUT_LOC)/8/MP2048dense.mtx > /dev/null
	@./res $(INPUT_LOC)/8/ModifiedParallel4096by4096dense_part_ 8 $(OUTPUT_LOC)/8/MP4096dense.mtx > /dev/null

	@./res $(INPUT_LOC)/16/ModifiedParallel4by4dense_part_ 16 $(OUTPUT_LOC)/16/MP4dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel16by16dense_part_ 16 $(OUTPUT_LOC)/16/MP16dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel32by32dense_part_ 16 $(OUTPUT_LOC)/16/MP32dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel64by64dense_part_ 16 $(OUTPUT_LOC)/16/MP64dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel128by128dense_part_ 16 $(OUTPUT_LOC)/16/MP128dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel256by256dense_part_ 16 $(OUTPUT_LOC)/16/MP256dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel512by512dense_part_ 16 $(OUTPUT_LOC)/16/MP512dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel1024by1024dense_part_ 16 $(OUTPUT_LOC)/16/MP1024dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel2048by2048dense_part_ 16 $(OUTPUT_LOC)/16/MP2048dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel4096by4096dense_part_ 16 $(OUTPUT_LOC)/16/MP4096dense.mtx > /dev/null
	@./res $(INPUT_LOC)/16/ModifiedParallel8192by8192dense_part_ 16 $(OUTPUT_LOC)/16/MP8192dense.mtx > /dev/null

	@./res $(INPUT_LOC)/32/ModifiedParallel4by4dense_part_ 32 $(OUTPUT_LOC)/32/MP4dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel16by16dense_part_ 32 $(OUTPUT_LOC)/32/MP16dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel32by32dense_part_ 32 $(OUTPUT_LOC)/32/MP32dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel64by64dense_part_ 32 $(OUTPUT_LOC)/32/MP64dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel128by128dense_part_ 32 $(OUTPUT_LOC)/32/MP128dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel256by256dense_part_ 32 $(OUTPUT_LOC)/32/MP256dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel512by512dense_part_ 32 $(OUTPUT_LOC)/32/MP512dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel1024by1024dense_part_ 32 $(OUTPUT_LOC)/32/MP1024dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel2048by2048dense_part_ 32 $(OUTPUT_LOC)/32/MP2048dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel4096by4096dense_part_ 32 $(OUTPUT_LOC)/32/MP4096dense.mtx > /dev/null
	@./res $(INPUT_LOC)/32/ModifiedParallel8192by8192dense_part_ 32 $(OUTPUT_LOC)/32/MP8192dense.mtx > /dev/null

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
