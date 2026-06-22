#include "block_gemm_cuda.h"

#include <cuda/cmath>

#define BLOCK_SIZE 16

__global__ void BlockGemmCUDAImpl(const float *a, const float *b, float *c, int n) {
    __shared__ float blockA[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float blockB[BLOCK_SIZE * BLOCK_SIZE];

    int threadX = threadIdx.x;
    int threadY = threadIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + threadX;
    int y = blockIdx.y * BLOCK_SIZE + threadY;

    float sum = 0.0f;
    for (int block = 0; block < gridDim.x; ++block) {
        blockA[threadY * BLOCK_SIZE + threadX] = a[y * n + block * BLOCK_SIZE + threadX];
        blockB[threadY * BLOCK_SIZE + threadX] = b[(block * BLOCK_SIZE + threadY) * n + x];
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += blockA[threadY * BLOCK_SIZE + k] * blockB[k * BLOCK_SIZE + threadX];
        }
        __syncthreads();
    }
    c[y * n + x] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const int size = a.size();
    const int bSize = size * sizeof(float);

    float* cHostPtr = nullptr;
    float* aDevicePtr = nullptr;
    float* bDevicePtr = nullptr;
    float* cDevicePtr = nullptr;

    cudaMalloc(&aDevicePtr, bSize);
    cudaMalloc(&bDevicePtr, bSize);
    cudaMalloc(&cDevicePtr, bSize);

    cudaMemcpy(aDevicePtr, a.data(), bSize, cudaMemcpyHostToDevice);
    cudaMemcpy(bDevicePtr, b.data(), bSize, cudaMemcpyHostToDevice);

    constexpr int nThreads = BLOCK_SIZE;
    int blocks = n / nThreads;
    dim3 threadsDim(nThreads, nThreads);
    dim3 blocksDim(blocks, blocks);
    BlockGemmCUDAImpl<<<blocksDim, threadsDim>>>(aDevicePtr, bDevicePtr, cDevicePtr, n);

    std::vector<float> c(size);
    cHostPtr = c.data();

    cudaDeviceSynchronize();
    cudaMemcpy(cHostPtr, cDevicePtr, bSize, cudaMemcpyDeviceToHost);
    cudaFree(aDevicePtr);
    cudaFree(bDevicePtr);
    cudaFree(cDevicePtr);

    return c;
}
