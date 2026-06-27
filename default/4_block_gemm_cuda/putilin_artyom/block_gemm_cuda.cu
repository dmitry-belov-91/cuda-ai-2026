#include "block_gemm_cuda.h"

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda/cmath>
#include <cmath>
#include <cstdlib>
#include <chrono>

#define TILE_DIM 16

__global__ void tiled_matrix_mul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int n, int numTiles)
{
    __shared__ float tileA[TILE_DIM][TILE_DIM];
    __shared__ float tileB[TILE_DIM][TILE_DIM];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for (int iBlock = 0; iBlock < numTiles; ++iBlock)
    {
        int colA = iBlock * TILE_DIM + tx;
        if (row < n && colA < n)
            tileA[ty][tx] = A[row * n + colA];
        else
            tileA[ty][tx] = 0.0f;

        int rowB = iBlock * TILE_DIM + ty;
        if (rowB < n && col < n)
            tileB[ty][tx] = B[rowB * n + col];
        else
            tileB[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_DIM; ++i)
        {
            sum += tileA[ty][i] * tileB[i][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n)
    {
        C[row * n + col] = sum;
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

    int numTiles = (n + TILE_DIM - 1) / TILE_DIM;

    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 blocksPerGrid(numTiles, numTiles);

    tiled_matrix_mul_kernel<<<blocksPerGrid, threadsPerBlock>>>(a_ptr, b_ptr, c_ptr, n, numTiles);

    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), c_ptr, mtxSize, cudaMemcpyDeviceToHost);

    cudaFree(a_ptr);
    cudaFree(b_ptr);
    cudaFree(c_ptr);

    return c;
}
