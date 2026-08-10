#include <cuda_runtime.h>

__global__
// m1 and m2 are input matrices
// m3 is the output matrix
// R represents how many rows a matrix has
// C represents how many columns a matrix has
// K represents the width of m1
void matrixMult(float* m1, float* m2, float* m3, unsigned int R, unsigned int C, int K)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < R) && (col < C))
    {
        // storage for computation
        float Fvalue = 0;
        for (int k = 0; k < K; ++k)
        {
            Fvalue += m1[row * K + k] * m2[k * C + col];
        }
        m3[row * C + col] = Fvalue;
    }
}
