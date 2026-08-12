#include <cuda_runtime.h>

#define TM 4
#define TN 4
#define TILE_WIDTH 64
#define STRIDE (TILE_WIDTH / TM)

__global__

// m1 and m2 are input matrices
// m3 is the output matrix
// R is the rows in the matrix
// C is the columns in the matrix
// K is the width of the matrix
void tiledMult(float* m1, float* m2, float* m3, unsigned int R, unsigned int C, int K)
{
    // declares float variables in shared memory with STRIDE * STRIDE = 64 positions
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; // which tile-row
    int by = blockIdx.y; // which tile-column
    int tx = threadIdx.x; // which column inside the tile
    int ty = threadIdx.y; // which row inside the tile

    int row = by * (TILE_WIDTH) + ty;
    int col = bx * (TILE_WIDTH) + tx;
    float acc[TM][TN] = {0};

    for (int i = 0; i < (K + TILE_WIDTH - 1)/TILE_WIDTH; ++i)
    {
        // loading matrix elements into shared memory
        for (int dy = 0; dy < TM; ++dy)
        {
            for (int dx = 0; dx < TN; ++dx)
            {
                if ((row + dy*STRIDE < R) && (i * TILE_WIDTH + tx + dx*STRIDE) < K)
                {
                    Mds[ty + dy*STRIDE][tx + dx*STRIDE] = m1[(row + dy*STRIDE) * K + i*TILE_WIDTH + tx + dx*STRIDE];
                }
                else
                {
                    Mds[ty + dy*STRIDE][tx + dx*STRIDE] = 0.0f;
                }
                if ((col + dx*STRIDE < C) && (i * TILE_WIDTH + ty + dy*STRIDE) < K)
                {
                    Nds[ty + dy*STRIDE][tx + dx*STRIDE] = m2[(i * TILE_WIDTH + ty + dy*STRIDE) * C + col + dx*STRIDE];
                }
                else
                {
                    Nds[ty + dy*STRIDE][tx + dx*STRIDE] = 0.0f;
                }
            }
        }
        __syncthreads();

        // computing final value
        for (int j = 0; j < TILE_WIDTH; ++j)
        {
            float a[TM];
            float b[TN];
            #pragma unroll
            for (int y = 0; y < TM; ++y)
            {
                a[y] = Mds[ty + y*STRIDE][j];
            }

            #pragma unroll
            for (int q = 0; q < TN; ++q)
            {
                b[q] = Nds[j][tx + q*STRIDE];
            }

            #pragma unroll
            for (int x = 0; x < TM; ++x)
            {
                #pragma unroll
                for (int u = 0; u < TN; ++u)
                {
                    acc[x][u] += a[x] * b[u];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i)
    {
        #pragma unroll
        for (int j = 0; j < TN; ++j)
        {
            if (((row + i*STRIDE) < R) && ((col + j*STRIDE) < C))
            {
                m3[(row + (i*STRIDE)) * C + col + j*STRIDE] = acc[i][j];
            }
        }
    }
}
