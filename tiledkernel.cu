#include <cuda_runtime.h>

#define TILE_WIDTH 32

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
    int tx = threadIdx.x; // which column inside the tile
    int ty = threadIdx.y; // which row inside the tile

    int row = by * (TILE_WIDTH) + ty;
    int col = bx * (TILE_WIDTH) + tx;
    float Fvalue00 = 0;
    float Fvalue01 = 0;
    float Fvalue10 = 0;
    float Fvalue11 = 0;

    for (int i = 0; i < (K + TILE_WIDTH - 1)/TILE_WIDTH; ++i)
    {
        // loading matrix elements into shared memory
        for (int dy = 0; dy < 2; ++dy)
        {
            for (int dx = 0; dx < 2; ++dx)
            {
                if ((row + dy*16 < R) && (i * TILE_WIDTH + tx + dx*16) < K)
                {
                    Mds[ty + dy*16][tx + dx*16] = m1[(row + dy*16) * K + i*TILE_WIDTH + tx + dx*16];
                }
                else
                {
                    Mds[ty + dy*16][tx + dx*16] = 0.0f;
                }
                if ((col + dx*16 < C) && (i * TILE_WIDTH + ty + dy*16) < K)
                {
                    Nds[ty + dy*16][tx + dx*16] = m2[(i * TILE_WIDTH + ty + dy*16) * C + col + dx*16];
                }
                else
                {
                    Nds[ty + dy*16][tx + dx*16] = 0.0f;
                }
            }
        }
        __syncthreads();

        // computing final value
        for (int j = 0; j < TILE_WIDTH; ++j)
        {
            float a0 = Mds[ty][j];
            float b0 = Nds[j][tx];
            float a1 = Mds[ty + 16][j];
            float b1 = Nds[j][tx + 16];

            Fvalue00 += a0 * b0;
            Fvalue01 += a0 * b1;
            Fvalue10 += a1 * b0;
            Fvalue11 += a1 * b1;
        }
        __syncthreads();
    }
    if ((row < R) && (col < C))
    {
        m3[row * C + col] = Fvalue00;
    }
    if ((row < R) && (col+16 < C))
    {
        m3[row * C + col+16] = Fvalue01;
    }
    if ((row+16 < R) && (col < C))
    {
        m3[(row+16) * C + col] = Fvalue10;
    }
    if ((row+16 < R) && (col+16 < C))
    {
        m3[(row+16) * C + col+16] = Fvalue11;
    }
}
