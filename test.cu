// Test + benchmark harness for gpukernel.cu
//
// The kernels stay yours. This grades them and times them.
//
// Build:  make
// Run:    ./test
//
// ---------------------------------------------------------------------------
// TURN VARIANTS ON AS YOU WRITE THEM. Flip 0 to 1.
// ---------------------------------------------------------------------------
#define TEST_TILED      1   // needs: tiledMult(...) in tiledkernel.cu
#define TEST_COARSENED  0   // needs: matrixMultCoarsened(...)
#define TEST_CUBLAS     1   // the ceiling. NVIDIA's own kernel, for reference.

// Must match the values you use inside your kernels.
#define COARSE_BLOCK    16
#define COARSE_FACTOR   4

// tiledkernel.cu: 16x16 threads, each owning a 2x2 patch -> 32x32 output tile.
#define TILED_BLOCK     16
#define TILED_OUT_TILE  64   // must match TILE_WIDTH in tiledkernel.cu

// ---------------------------------------------------------------------------
// Expected kernel signatures, all identical to your naive one:
//
//   __global__ void matrixMultTiled    (float* m1, float* m2, float* m3,
//                                       unsigned int R, unsigned int C, int K);
//   __global__ void matrixMultCoarsened(float* m1, float* m2, float* m3,
//                                       unsigned int R, unsigned int C, int K);
//
// Launch configs this harness uses:
//   naive      block(16,16)            grid(ceil(C/16), ceil(R/16))
//   tiled      block(TW,TW)            grid(ceil(C/TW), ceil(R/TW))
//   coarsened  block(TW,TW)            grid(ceil(C/(TW*CF)), ceil(R/TW))
//              -> each thread computes COARSE_FACTOR output columns
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#if TEST_CUBLAS
#include <cublas_v2.h>
#endif

#include "gpukernel.cu"

#if TEST_TILED
#include "tiledkernel.cu"
#endif

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            printf("CUDA error %s at %s:%d\n",                                 \
                   cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

enum Variant { NAIVE, TILED, COARSENED, CUBLAS };

static const char* variantName(Variant v)
{
    switch (v) {
        case NAIVE:     return "naive";
        case TILED:     return "tiled";
        case COARSENED: return "coarsened";
        case CUBLAS:    return "cuBLAS";
    }
    return "?";
}

static int variantEnabled(Variant v)
{
    switch (v) {
        case NAIVE:     return 1;
        case TILED:     return TEST_TILED;
        case COARSENED: return TEST_COARSENED;
        case CUBLAS:    return TEST_CUBLAS;
    }
    return 0;
}

// ---------------------------------------------------------------------------

#if TEST_CUBLAS
static cublasHandle_t g_blas;
#endif

static void launch(Variant v, const float* dA, const float* dB, float* dC,
                   int R, int C, int K)
{
    unsigned uR = (unsigned)R, uC = (unsigned)C;

    if (v == NAIVE) {
        dim3 block(16, 16);
        dim3 grid((C + 15) / 16, (R + 15) / 16);
        matrixMult<<<grid, block>>>((float*)dA, (float*)dB, dC, uR, uC, K);
    }
#if TEST_TILED
    else if (v == TILED) {
        dim3 block(TILED_BLOCK, TILED_BLOCK);
        dim3 grid((C + TILED_OUT_TILE - 1) / TILED_OUT_TILE,
                  (R + TILED_OUT_TILE - 1) / TILED_OUT_TILE);
        tiledMult<<<grid, block>>>((float*)dA, (float*)dB, dC, uR, uC, K);
    }
#endif
#if TEST_COARSENED
    else if (v == COARSENED) {
        int cols = COARSE_BLOCK * COARSE_FACTOR;
        dim3 block(COARSE_BLOCK, COARSE_BLOCK);
        dim3 grid((C + cols - 1) / cols,
                  (R + COARSE_BLOCK - 1) / COARSE_BLOCK);
        matrixMultCoarsened<<<grid, block>>>((float*)dA, (float*)dB, dC, uR, uC, K);
    }
#endif
#if TEST_CUBLAS
    else if (v == CUBLAS) {
        // cuBLAS is column-major. Computing B^T * A^T in its view gives us
        // the row-major C we want, with no transposes and no copies.
        const float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N,
                    C, R, K,
                    &alpha,
                    dB, C,
                    dA, K,
                    &beta,
                    dC, C);
    }
#endif
}

// ---------------------------------------------------------------------------

static void matrixMultCPU(const float* A, const float* B, float* C,
                          int R, int C_cols, int K)
{
    for (int row = 0; row < R; row++) {
        for (int col = 0; col < C_cols; col++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * C_cols + col];
            }
            C[row * C_cols + col] = sum;
        }
    }
}

static void fill(float* p, int n, unsigned seed)
{
    unsigned s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = (float)((s >> 16) % 17) - 8.0f;
    }
}

static int runCase(Variant v, int R, int C_cols, int K, const char* label)
{
    size_t bytesA = (size_t)R * K * sizeof(float);
    size_t bytesB = (size_t)K * C_cols * sizeof(float);
    size_t bytesC = (size_t)R * C_cols * sizeof(float);

    float* hA   = (float*)malloc(bytesA);
    float* hB   = (float*)malloc(bytesB);
    float* hC   = (float*)malloc(bytesC);
    float* hRef = (float*)malloc(bytesC);

    fill(hA, R * K, 1u);
    fill(hB, K * C_cols, 2u);
    matrixMultCPU(hA, hB, hRef, R, C_cols, K);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, bytesC));

    launch(v, dA, dB, dC, R, C_cols, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));

    int   bad = 0, br = -1, bc = -1;
    float got = 0.0f, want = 0.0f;
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C_cols; c++) {
            float g = hC[r * C_cols + c], w = hRef[r * C_cols + c];
            if (fabsf(g - w) > 1e-3f * (fabsf(w) + 1.0f)) {
                if (bad == 0) { br = r; bc = c; got = g; want = w; }
                bad++;
            }
        }
    }

    int total = R * C_cols;
    if (bad == 0) {
        printf("  PASS  %-22s  %dx%d @ %dx%d\n", label, R, K, K, C_cols);
    } else {
        printf("  FAIL  %-22s  %dx%d @ %dx%d  (%d/%d wrong)\n",
               label, R, K, K, C_cols, bad, total);
        printf("        first bad element C[%d][%d]: got %g, want %g\n",
               br, bc, got, want);
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC); free(hRef);
    return bad == 0;
}

// Returns milliseconds per call, averaged.
static float benchmark(Variant v, int R, int C_cols, int K)
{
    // The 4090 idles at 210 MHz and boosts to ~3135. A fixed warmup iteration
    // count is not enough for a fast kernel, so warm up by elapsed TIME, then
    // take the best of several timed reps (noise only ever adds time).
    const float WARM_MS = 400.0f;
    const int ITERS = 20, REPS = 3;

    size_t bytesA = (size_t)R * K * sizeof(float);
    size_t bytesB = (size_t)K * C_cols * sizeof(float);
    size_t bytesC = (size_t)R * C_cols * sizeof(float);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMemset(dA, 0, bytesA));
    CUDA_CHECK(cudaMemset(dB, 0, bytesB));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float warm = 0.0f;
    CUDA_CHECK(cudaEventRecord(start));
    do {
        for (int i = 0; i < 5; i++) launch(v, dA, dB, dC, R, C_cols, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&warm, start, stop));
    } while (warm < WARM_MS);

    float ms = 1e30f;
    for (int r = 0; r < REPS; r++) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < ITERS; i++) launch(v, dA, dB, dC, R, C_cols, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float rep = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&rep, start, stop));
        if (rep < ms) ms = rep;
    }

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return ms / ITERS;
}

int main(void)
{
#if TEST_CUBLAS
    cublasCreate(&g_blas);
#endif

    Variant all[] = { NAIVE, TILED, COARSENED, CUBLAS };
    const int NV = 4;

    // ---------------- correctness ----------------
    printf("\n=== correctness ===\n");
    int pass = 0, total = 0;
    for (int i = 0; i < NV; i++) {
        Variant v = all[i];
        if (!variantEnabled(v)) continue;
        printf("\n%s\n", variantName(v));
        total++; pass += runCase(v, 64, 64, 64, "square, aligned");
        total++; pass += runCase(v, 67, 67, 67, "square, ragged");
        total++; pass += runCase(v, 67, 43, 51, "non-square");
        total++; pass += runCase(v, 32, 96, 16, "wide output");
        total++; pass += runCase(v, 96, 32, 80, "tall output");
        total++; pass += runCase(v,  1, 64, 64, "single row");
        total++; pass += runCase(v, 64,  1, 64, "single column");
    }
    printf("\n%d/%d cases passed\n", pass, total);

    if (pass != total) {
        printf("\nskipping benchmark until everything is correct.\n\n");
        return 1;
    }

    // ---------------- performance ----------------
    const int N = 2048;
    double flop = 2.0 * N * N * N;

    printf("\n=== performance (%dx%d @ %dx%d) ===\n\n", N, N, N, N);
    printf("  %-12s %10s %12s %10s\n", "kernel", "ms", "GFLOP/s", "vs naive");
    printf("  %-12s %10s %12s %10s\n", "------", "--", "-------", "--------");

    float naiveMs = 0.0f;
    for (int i = 0; i < NV; i++) {
        Variant v = all[i];
        if (!variantEnabled(v)) continue;
        float ms = benchmark(v, N, N, N);
        if (v == NAIVE) naiveMs = ms;
        double gflops = flop / (ms * 1.0e6);
        printf("  %-12s %10.3f %12.1f %9.2fx\n",
               variantName(v), ms, gflops, naiveMs / ms);
    }
    printf("\n");

#if TEST_CUBLAS
    cublasDestroy(g_blas);
#endif
    return 0;
}
