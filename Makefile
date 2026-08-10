NVCC  := nvcc
ARCH  := -arch=sm_89
FLAGS := -O2 $(ARCH)

test: test.cu gpukernel.cu
	$(NVCC) $(FLAGS) -o test test.cu

run: test
	./test

clean:
	rm -f test

.PHONY: run clean
