#include "block_gemm_cuda.h"

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda/cmath>
#include <cmath>
#include <cstdlib>
#include <chrono>


#define TILE_DIM 16

__global__ void tiled_matrix_mul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N)
{
    __shared__ float tileA[TILE_DIM][TILE_DIM];
    __shared__ float tileB[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    float sum = 0.0f;
    int numTiles = (N + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < numTiles; ++t)
    {
        if (row < N && (t * TILE_DIM + tx) < N)
        {
            tileA[ty][tx] = A[row * N + (t * TILE_DIM + tx)];
        }
        else
        {
            tileA[ty][tx] = 0.0f;
        }

        if (col < N && (t * TILE_DIM + ty) < N)
        {
            tileB[ty][tx] = B[(t * TILE_DIM + ty) * N + col];
        }
        else
        {
            tileB[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k)
        {
            sum += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N)
    {
        C[row * N + col] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n)
{
    size_t N = n * n;
    size_t mtxSize = N * sizeof(float);
    std::vector<float> c(N);

    float *a_ptr = nullptr;
    float *b_ptr = nullptr;
    float *c_ptr = nullptr;

    cudaMalloc(&a_ptr, mtxSize);
    cudaMalloc(&b_ptr, mtxSize);
    cudaMalloc(&c_ptr, mtxSize);

    cudaMemcpy(a_ptr, a.data(), mtxSize, cudaMemcpyHostToDevice);
    cudaMemcpy(b_ptr, b.data(), mtxSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 blocksPerGrid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    tiled_matrix_mul_kernel<<<threadsPerBlock, blocksPerGrid>>>(a_ptr, b_ptr, c_ptr, n);

    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), c_ptr, mtxSize, cudaMemcpyDeviceToHost);

    cudaFree(a_ptr);
    cudaFree(b_ptr);
    cudaFree(c_ptr);

    return c;
}
