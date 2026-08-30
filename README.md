# cuda-tiled-sgemm

A single-precision dense matrix multiplication kernel for the RTX 4090, written from scratch and optimized to 80% of cuBLAS throughput. Built after finishing Chapter 6 of *Programming Massively Parallel Processors*.

## 1. Overview

This repo contains three kernels computing C = A * B, all at different speeds:

| Kernel | File | What It Is |
|---|---|---|
| CPU Reference | `cpukernel.c` | Triple loop, used to grade the GPU results |
| Naive | `gpukernel.cu` | One thread per output element, all reads from global memory |
| Tiled | `tiledkernel.cu` | 128x128 block tiles, 8x8 register tiles, double-buffered shared memory |

cuBLAS is compiled into the harness as a fourth kernel. It is the ceiling these kernels are measured against.

## 2. Quick Start

```
make
./test
```

This requires CUDA and an sm_89 card. The arch is set in the Makefile.

The harness runs correctness first, then performance.

## 3. Results

Measured on an RTX 4090, CUDA 13.3, driver 610.57.04, at 2048x2048 by 2048x2048.

| Kernel | ms | GFLOP/s | vs Naive | vs cuBLAS |
|---|---|---|---|---|
| Naive | 3.371 | 5097 | 1.00x | 9.2% |
| Tiled | 0.389 | 44110 | 8.65x | 79.8% |
| cuBLAS | 0.311 | 55297 | 10.85x | 100% |

Correctness: 21/21 cases for all three kernels.

### 3a. How It Got There

The kernel went through multiple commits:

| Step | GFLOP/s | % of cuBLAS | What Changed |
|---|---|---|---|
| Naive | 5097 | 9% | one thread, one output, every read from global |
| Tiled | See Note | | shared memory tiles, reported as 3.33x over naive |
| 2x2 Register Tile | 13600 | 25% | each thread owns a 2x2 patch | 
| 4x4 Register Tile | 26100 | 47% | each thread owns a 4x4 patch |
| 8x8 Register Tile | 27600 | 50% | BK decoupled from TILE_WIDTH |
| Contiguous Strips | 42600 | 77% | strip ownership, bank-conflict-free shared loads |
| Double Buffering | 44110 | 80% | prefetch batch i+1 while computing batch i |

Note on the tiled row: that commit recorded a ratio, not a throughput. The 2x2 step that follows it measured lower than the ratio implies, so the first two rows should be read as the shape of the climb and not as two directly comparable data points.

It's also easy to notice that the jump from 4x4 to 8x8 was relatively minimal. At this stage, the kernel became limited by shared memory bandwidth.

### 3b. Occupancy

`ptxas` reports 120 registers, 0 bytes spilled, 16384 bytes of shared memory, 256 threads per block. That works out to 2 blocks resident per SM, or 33% occupancy.

This is intentional and not a defect. The kernel holds a 64-element accumulator per thread in registers, which makes the arithmetic intensity high enough to hit 80%. Trading those registers away for more resident warps would lower throughput.

## 4. Architecture

### 4a. The Tiling Hierarchy

```
    output matrix C             2048 x 2048 = 4,194,304 elements
        block tile              128 x 128   = 16,384 elements   (16x16 grid of blocks)
            thread tile         8 x 8       = 64 elements       (16x16 = 256 threads)
```

Each block walks the K dimension in batches of BK = 8. Per batch it stages a 128x8 slice of A and an 8x128 slice of B into shared memory, then every thread runs 8 outer products against its own 8x8 accumulator.

A is transposed on the way into shared memory, from `[row][k]` to `[k][row]`, so that both operands are read down a column of the shared tile during the inner product.

### 4b. Bank Conflicts

Shared memory has 32 banks. The fix that got the kernel from 50% to 76% was making the stride between the start of consecutive strips match the read width: each thread reads 4 floats at a time, so the ownership pattern deals rows in steps of 4, and the second strip starts 64 elements away from the first. That keeps the 32 lanes of a warp on 32 distinct banks.

The reason an explicit `reinterpret_cast<float4*>` and `__launch_bounds__` were not added is because nvcc auto-vectorizes the contiguous loads into `LDS.128`. When I added both lines, the kernel actually performed slower.

### 4c. Double Buffering

Shared memory is declared as `[2][BK][TILE_WIDTH]`. Batch `i` computes out of buffer `i % 2` while the global loads for batch `i+1` land in the other buffer. A prologue primes buffer 0 before the main loop so the first iteration has data to work on. This removes one of the two `__syncthreads()` per batch and lets the memory pipeline overlap the math.

## 5. Verification

`test.cu` grades every kernel against a CPU reference computed in double precision, across seven shapes chosen to break tiling assumptions:

| Case | Shape | What It Catches | 
|---|---|---| 
| Square, Aligned | 64x64 By 64x64 | Baseline |
| Square, Ragged | 67x67 By 67x67 | Partial tiles on all three dimensions |
| Non-Square | 67x51 By 51x43 | R, C, and K all different and all ragged |
| Wide Output | 32x16 By 16x96 | C much larger than R |
| Tall Output | 96x80 By 80x32 | R much larger than C |
| Single Row | 1x64 By 64x64 | Degenerate R |
| Single Column | 64x64 By 64x1 | Degenerate C |

The three kernels times seven shapes is the 21/21. The ragged cases show the bounds guards protect against out-of-bounds reads and writes.

## 6. Limitations

- Single precision only. No tensor cores, so 80% of cuBLAS here means 80% of cuBLAS SGEMM, not of what the hardware can do in TF32 or FP16.
- No warptiling, which is where the next 20 points lie.
- Alpha and beta are not implemented. This is just C = A * B, not the full BLAS GEMM signature.
- Arch is hardcoded to sm_89 in the Makefile.
- Row-major only.

## 7. Authorship and References

| Mine | AI-Assisted: Tooling, Instrumentation, Measurement |
|---|---|
| `cpukernel.c` | `test.cu` |
| `gpukernel.cu` | `Makefile` |
| `tiledkernel.cu` | README |

Every kernel in this repo is mine, completely. The tests and benchmarks were built with AI assistance.

- Hwu, Kirk, and El Hajj, *Programming Massively Parallel Processors*, 5th Edition.
