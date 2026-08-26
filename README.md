# cuda-tiled-sgemm

A single-precision dense matrix multiply kernel for the RTX 4090, written from
scratch and optimized to 80% of cuBLAS throughput. Built while working through
Chapters 4 to 6 of *Programming Massively Parallel Processors*.

## 1. Overview

Three kernels compute the same thing, C = A * B, at very different speeds:

| Kernel | File | What it is |
|---|---|---|
| CPU reference | `cpukernel.c` | Triple loop, used to grade the GPU results |
| Naive | `gpukernel.cu` | One thread per output element, all reads from global memory |
| Tiled | `tiledkernel.cu` | 128x128 block tiles, 8x8 register tiles, double-buffered shared memory |

cuBLAS is compiled into the harness as a fourth kernel. It is not a target to
beat, it is the ceiling to measure against.

## 2. Quick Start

Requires CUDA and an sm_89 card. The arch is set in the Makefile.

```
make
./test
```

The harness runs correctness first, then performance. Nothing is timed until
every shape passes.

## 3. Results

Measured on an RTX 4090, CUDA 13.3, driver 610.57.04, at 2048x2048 @ 2048x2048.

| Kernel | ms | GFLOP/s | vs naive | vs cuBLAS |
|---|---|---|---|---|
| naive | 3.371 | 5097 | 1.00x | 9.2% |
| tiled | 0.389 | 44110 | 8.65x | 79.8% |
| cuBLAS | 0.311 | 55297 | 10.85x | 100% |

Correctness: 21/21 cases, all three kernels, every run.

### 3a. How It Got There

Each row is one commit. The kernel was measured after every change, and no
change was kept unless the number moved.

| Step | GFLOP/s | % of cuBLAS | What changed |
|---|---|---|---|
| naive | 5097 | 9% | one thread, one output, every read from global |
| tiled | see note | | shared memory tiles, reported as 3.33x over naive |
| 2x2 register tile | 13600 | 25% | each thread owns a 2x2 patch |
| 4x4 register tile | 26100 | 47% | each thread owns a 4x4 patch |
| 8x8 register tile | 27600 | 50% | BK decoupled from TILE_WIDTH |
| contiguous strips | 42600 | 76% | strip ownership, bank-conflict-free shared loads |
| double buffering | 44110 | 80% | prefetch batch i+1 while computing batch i |

Note on the tiled row: that commit recorded a ratio, not a throughput, so it is
left as reported rather than back-computed. The 2x2 step that follows it
measured lower than the ratio implies, so the first two rows should be read as
the shape of the climb and not as two directly comparable data points. Every
row from 2x2 down is a measured GFLOP/s figure at the same 2048 benchmark.

The jump from 4x4 to 8x8 returned only 6%. That is where register tiling ran
out: the kernel had stopped being register-bound and become limited by shared
memory bandwidth. The next two steps attack shared memory directly, and they
are where the remaining 30 points came from.

### 3b. Occupancy

`ptxas` reports 120 registers, 0 bytes spilled, 16384 bytes of shared memory,
256 threads per block. That works out to 2 blocks resident per SM, or 33%
occupancy.

This is intentional and it is not a defect. The kernel holds a 64-element
accumulator per thread in registers, which is exactly what makes the arithmetic
intensity high enough to hit 80%. Trading those registers away for more
resident warps would lower throughput, not raise it.

## 4. Architecture

### 4a. The Tiling Hierarchy

```
  output matrix C         2048 x 2048  = 4,194,304 elements
    block tile            128  x 128   = 16,384 elements    (16x16 grid of blocks)
      thread tile         8    x 8     = 64 elements        (16x16 = 256 threads)
```

Each block walks the K dimension in batches of BK = 8. Per batch it stages a
128x8 slice of A and an 8x128 slice of B into shared memory, then every thread
runs 8 outer products against its own 8x8 accumulator.

A is transposed on the way into shared memory, from `[row][k]` to `[k][row]`,
so that both operands are read down a column of the shared tile during the
inner product.

### 4b. Bank Conflicts

Shared memory has 32 banks. The fix that got the kernel from 50% to 76% was
making the stride between the start of consecutive strips match the read width:
each thread reads 4 floats at a time, so the ownership pattern deals rows in
steps of 4, and the second strip starts 64 elements away from the first. That
keeps the 32 lanes of a warp on 32 distinct banks.

Note that nvcc auto-vectorizes the contiguous loads into `LDS.128` on its own.
Writing explicit `reinterpret_cast<float4*>` and adding `__launch_bounds__` were
both tried and both measured *slower*. They are deliberately not in the code.

### 4c. Double Buffering

Shared memory is declared as `[2][BK][TILE_WIDTH]`. Batch `i` computes out of
buffer `i % 2` while the global loads for batch `i+1` land in the other buffer.
A prologue primes buffer 0 before the main loop so the first iteration has data
to work on. This removes one of the two `__syncthreads()` per batch and lets the
memory pipeline overlap the math.

## 5. Verification

`test.cu` grades every kernel against a CPU reference computed in double
precision, across seven shapes chosen to break tiling assumptions:

| Case | Shape | What it catches |
|---|---|---|
| square, aligned | 64x64 @ 64x64 | baseline |
| square, ragged | 67x67 @ 67x67 | partial tiles on all three dimensions |
| non-square | 67x51 @ 51x43 | R, C, and K all different and all ragged |
| wide output | 32x16 @ 16x96 | C much larger than R |
| tall output | 96x80 @ 80x32 | R much larger than C |
| single row | 1x64 @ 64x64 | degenerate R |
| single column | 64x64 @ 64x1 | degenerate C |

Three kernels times seven shapes is the 21/21. The ragged cases are the ones
that matter: they are what the boundary guards in the fill loops exist for.

## 6. Limitations

- Single precision only. No tensor cores, so 80% of cuBLAS here means 80% of
  cuBLAS SGEMM, not of what the hardware can do in TF32 or FP16.
- No warptiling, which is the standard next step and the reason the last 20
  points are still on the table.
- Alpha and beta are not implemented. This is C = A * B, not the full BLAS
  GEMM signature.
- Arch is hardcoded to sm_89 in the Makefile.
- Row-major only, no transpose flags.

## 7. Authorship and References

| Mine | Tooling, built with AI assistance |
|---|---|
| cpukernel.c | test.cu |
| gpukernel.cu | Makefile |
| tiledkernel.cu | This README |

Every kernel in this repo is mine, written line by line. The harness grades
them and times them, and was built with AI assistance so that the kernels
themselves stayed my work.

- Hwu, Kirk, and El Hajj, *Programming Massively Parallel Processors*, 5th edition.
- NVIDIA CUDA C++ Programming Guide.
