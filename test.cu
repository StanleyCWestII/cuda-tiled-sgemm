// Test harness for gpukernel.cu
// The kernel stays yours. This is just a tool that tells you when it's right.
//
// Build:  make
// Run:    ./test

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#include "gpukernel.cu"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            printf("CUDA error %s at %s:%d\n",                                 \
                   cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// Reference implementation, straight from your cpukernel.c
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
    // deterministic, small values so float error stays tiny
    unsigned s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = (float)((s >> 16) % 17) - 8.0f;   // -8 .. 8
    }
}

// Runs one shape. Returns 1 on pass, 0 on fail.
static int runCase(int R, int C_cols, int K, const char* label)
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

    dim3 block(16, 16);
    dim3 grid((C_cols + block.x - 1) / block.x,
              (R      + block.y - 1) / block.y);

    matrixMult<<<grid, block>>>(dA, dB, dC, R, C_cols, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));

    int    bad       = 0;
    int    firstBadR = -1, firstBadC = -1;
    float  got = 0.0f, want = 0.0f;

    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C_cols; c++) {
            float g = hC[r * C_cols + c];
            float w = hRef[r * C_cols + c];
            if (fabsf(g - w) > 1e-3f * (fabsf(w) + 1.0f)) {
                if (bad == 0) {
                    firstBadR = r; firstBadC = c; got = g; want = w;
                }
                bad++;
            }
        }
    }

    int total = R * C_cols;
    if (bad == 0) {
        printf("  PASS  %-22s  %dx%d @ %dx%d  (%d/%d elements)\n",
               label, R, K, K, C_cols, total, total);
    } else {
        printf("  FAIL  %-22s  %dx%d @ %dx%d  (%d/%d wrong)\n",
               label, R, K, K, C_cols, bad, total);
        printf("        first bad element C[%d][%d]: got %g, want %g\n",
               firstBadR, firstBadC, got, want);
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC); free(hRef);
    return bad == 0;
}

int main(void)
{
    printf("\nsgemm kernel tests\n\n");

    int pass = 0, total = 0;

    // square, dims divisible by block size. this one is the liar:
    // it passes even with the strides confused, because R == C == K
    total++; pass += runCase(64, 64, 64, "square, aligned");

    // square but not a multiple of 16, exercises the bounds guard
    total++; pass += runCase(67, 67, 67, "square, ragged");

    // the real tests. R, C, K all different, so every stride must be right
    total++; pass += runCase(67, 43, 51, "non-square");
    total++; pass += runCase(32, 96, 16, "wide output");
    total++; pass += runCase(96, 32, 80, "tall output");
    total++; pass += runCase(1,  64, 64, "single row");
    total++; pass += runCase(64,  1, 64, "single column");

    printf("\n%d/%d cases passed\n\n", pass, total);
    return (pass == total) ? 0 : 1;
}
