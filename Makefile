NVCC  := nvcc
ARCH  := -arch=sm_89
FLAGS := -O2 $(ARCH)
LIBS  := -lcublas

test: test.cu gpukernel.cu tiledkernel.cu
	$(NVCC) $(FLAGS) -o test test.cu $(LIBS)

run: test
	./test

clean:
	rm -f test

.PHONY: run clean
