#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__

// m1 and m2 are input matrices
// m3 is the output matrix
// R is the rows in the matrix
// C is the columns in the matrix
// K is the width of the matrix
void tiledMult(float* m1, float* m2, float* m3, unsigned int R, unsigned int C, int K)
{
    // declares float variables in shared memory with 16 * 16 = 64 positions
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; // which tile-row
    int by = blockIdx.y; // which tile-column
    int tx = threadIdx.x; // which row inside the tile
    int ty = threadIdx.y; // which column inside the tile

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float Fvalue = 0;

    for (int i = 0; i < (K + TILE_WIDTH - 1)/TILE_WIDTH; ++i)
    {
        // loading matrix elements into shared memory
        if ((row < R) && (i * TILE_WIDTH + tx) < K)
        {
            Mds[ty][tx] = m1[row * K + i * TILE_WIDTH + tx];
        }
        else
        {
            Mds[ty][tx] = 0.0f;
        }
        if ((col < C) && (i * TILE_WIDTH + ty) < K)
        {
            Nds[ty][tx] = m2[(i * TILE_WIDTH + ty) * C + col];
        }
        else
        {
            Nds[ty][tx] = 0.0f;
        }
        __syncthreads();

        // computing final value
        for (int j = 0; j < TILE_WIDTH; ++j)
        {
            Fvalue += Mds[ty][j] * Nds[j][tx];
        }
        __syncthreads();
    }
    if ((row < R) && (col < C))
    {
        m3[row * C + col] = Fvalue;
    }
}
