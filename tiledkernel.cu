#include <cuda_runtime.h> // brings in cuda tools

// The complete hierachy:
// Level 1: m3, the whole output matrix is 2048 x 2048 = 4,194,304 elements
// Level 2: one block is 128 x 128 = 16,384 elements
// Level 3: one thread is 8 x 8 = 64 elements
// This kernel is running 65,536 threads
// 4104304 / 65536 = 64 elements per thread

// This kernel computes m1 * m2 = m3
// m1 is R*K
// m2 is K*C
// m3 is R*C

// To be clear, C represents the columns in a matrix; R represents the rows in
// a matrix, and K represents the width of m1 and height of m2

#define TM 8 // rows of output this thread owns
#define TN 8 // cols of output this thread owns
#define BLOCK_DIM 16 // block is 16x16 = 256 threads
#define BK 8 // K-terms handled per batch
#define TILE_WIDTH 128 // output patch one BLOCK owns, = BLOCK_DIM * TM
#define STRIDE (TILE_WIDTH / TM) // STRIDE is the gap between the outputs one thread owns

// A batch is one trip through the outermost i loop. BK is how big a trip is.
// In this kernel, there are 1024 + 1024 = 2048 products. With BK = 8, we have
// 256 batches.

__global__ // declares the kernel to be global

// m1 and m2 are input matrices; m3 is the output matrix
void tiledMult(float* m1, float* m2, float* m3, unsigned int R, unsigned int C, int K)
{
    // Mds represents m1's shared memory and Nds represents m2's shared memory.
    // Most importantly, m1 is indexed as m1[row][k] while m2 is indexed as
    // m2[k][col]. However, m1 has to be transposed into [k][row] for later
    // computation.

    // Mds is 8 wide, 32 tall. Nds is 128 wide, 2 tall.
    __shared__ float Mds[BK][TILE_WIDTH]; // a shared memory array 8 * 128 = 1024 large
    __shared__ float Nds[BK][TILE_WIDTH]; // together, we have 2048 elements

    // In CUDA, x is horizontal and y is vertical. In a matrix, you write m[row][col],
    // which is vertical first. That is why row is built from by/ty and column is
    // built from bx/tx

    // Now, we have 2048 elements. Each block owns a 128x128 patch, so the grid is
    // 16x16 blocks. Inside each block, there are 16x16 = 256 threads.

    int bx = blockIdx.x; // picks a vertical band of columns OUTSIDE the block
    int by = blockIdx.y; // picks a horizontal band of rows OUTSIDE the block

    int tx = threadIdx.x; // both run 0-15. picks a column INSIDE the block
    int ty = threadIdx.y; // picks a row INSIDE the block

    int row = by * (TILE_WIDTH) + ty; // which block's band, then how far down inside it
    int col = bx * (TILE_WIDTH) + tx; // which block's band, then how far across inside it

    // adr is row * width + col. It represents the address of a thread inside its
    // block. It can be anything from 0 to 255.
    int adr = ty * BLOCK_DIM + tx;

    int aValue = adr % BK; // which VALUE to grab from m1
    int aRow = adr / BK; // which ROW of m1 to grab the value from

    int bValue = adr / TILE_WIDTH; // which VALUE to grab from m2
    int bCol = adr % TILE_WIDTH; // which COLUMN of m2 to grab it from

    float acc[TM][TN] = {0}; // stores all 64 outputs of each thread

    // "i" represents which BATCH. this loops over (2048 + 8 - 1)/8 = 256 times.
    for (int i = 0; i < (K + BK - 1)/BK; ++i)
    {
        // #pragma unroll tells the compiler to replace the loop with copies of its
        // body, one per iteration. this allows all variables to be placed in
        // registers rather than local memory

        // For future reference, each load is performed by the "fast index," which
        // is just the first index in the matrix. For m1 it is K, and for m2 it is C.
        // Our m1 has a K of 8, and our m2 has a C of 128.

        #pragma unroll
        // This loop fills the Mds array. We start by computing the first 0-7 k terms.
        // Remember, m1 is [row][k]. Row is 128, so we iterate through 128, jumping
        // by 32 each time. For s=0, values 0-31 get filled, for s=1, values 32-63
        // get filled, and so on. This is because we copy the chunk's width,
        // take a quarter of its height, and stamp it 4 times. This tells us the chunk's
        // step and cap. For m1 (width = 8 and height = 128), we say 128/4 is 32, so step
        // = 32, and the cap is 32x4, or 128.
        for (int s = 0; s < TILE_WIDTH; s+= (BLOCK_DIM*BLOCK_DIM)/BK)
        {
            // by*TILE_WIDTH gives you which group of 128 you start at. aRow + s tells
            // you how far down to go in that block. It gives an absolute row of m1, from
            // 0 to 2047
            int r = by*TILE_WIDTH + aRow + s;
            // gives you an absolute COLUMN. i*BK computes the trip, while aValue tells
            // you how far right to go
            int k = i*BK + aValue;
            // the right side, m1[r*K + k], fetches row r, column k of m1. the left side,
            // Mds[aValue][aRow + s] is an index into the Mds array but reads as [k][row].
            Mds[aValue][aRow + s] = (r < R && k < K) ? m1[r*K + k] : 0.0f;
        }

        // This loop filles the Nds array. We also start by computing the first 0-7
        // k terms. m2 is [k][row], so we index through BK first, jumping by 2 each
        // time. For s=0, values 0-1 get filled, for s=1, values 2-3 get filled,
        // and so on. Like previously, m2 (width = 128 and height = 8), we say 8/2 is
        // 2, so step = 2, and 2x4 is 8, so cap = 8.
        #pragma unroll
        for (int s = 0; s < BK; s += (BLOCK_DIM*BLOCK_DIM)/TILE_WIDTH)
        {
            // inverse of Mds. k grabs the absolute row
            int k = i*BK + bValue + s;
            // grabs the absolute column
            int c = bx*TILE_WIDTH + bCol;
            // writes row, col. does not swap like Mds that writes col, row
            Nds[bValue + s][bCol] = (k < K && c < C) ? m2[k*C + c] : 0.0f;
        }

        __syncthreads(); // makes sure each thread is finished loading its values

        // j runs 0-7 for 256 batches, loading 8 * 256 = 2048 values total
        for (int j = 0; j < BK; ++j)
        {
            float a[TM]; // an array of 8 input rows
            float b[TN]; // an array of 8 column rows
            #pragma unroll
            for (int y = 0; y < TM; ++y) // iterates over all float a elements
            {
                // Mds is stored at 8 rows of 128. j walks each row. ty is the
                // specific thread id. STRIDE multiplies by 16 on each iteration,
                // picking 8 elements out of the 128 elements on each row.
                a[y] = Mds[j][ty + y*STRIDE];
            }

            #pragma unroll
            for (int y = 0; y < TN; ++y)
            {
                // same function but it walks the column
                b[y] = Nds[j][tx + y*STRIDE];
            }

            #pragma unroll
            for (int q = 0; q < TM; ++q)
            {
                #pragma unroll
                for (int u = 0; u < TN; ++u)
                {
                    // loop iterates over [8][8] = 64 entries of acc and stores the
                    // computation of a and b in each slot.
                    acc[q][u] += a[q] * b[u];
                }
            }
        }
        __syncthreads(); // makes sure no threads wrap to i+1 before others finish
    }

    #pragma unroll
    // like before, walks [8][8] = 64 elements, writing every element the thread
    // computed to m3
    for (int i = 0; i < TM; ++i)
    {
        #pragma unroll
        for (int j = 0; j < TN; ++j)
        {
            // checks if the row and column are out-of-bounds
            if (((row + i*STRIDE) < R) && ((col + j*STRIDE) < C))
            {
                // final write to m3. (row + (i*STRIDE)) walks each of the thread's
                // output rows. (col + (j*STRIDE)) walks each of the thread's output
                // columns. Multiplying by C allows you to step past each column
                // into the actual value
                m3[(row + (i*STRIDE)) * C + (col + j*STRIDE)] = acc[i][j];
            }
        }
    }
}
