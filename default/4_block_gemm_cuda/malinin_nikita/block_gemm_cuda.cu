#include "block_gemm_cuda.h"
#include <cuda/cmath>

#define BLOCK_SIZE 16
#define BLOCK_SIZE_DOUBLE BLOCK_SIZE * BLOCK_SIZE

__global__ void BlockGemmCUDAImpl(const float *a, const float *b, float *c, int n) {
    __shared__ int blockA[BLOCK_SIZE_DOUBLE];
    __shared__ int blockB[BLOCK_SIZE_DOUBLE];

    int localRow = threadIdx.y;
    int localCol = threadIdx.x;
    int globalRow = localRow + blockIdx.y * blockDim.y;
    int globalCol = localCol + blockIdx.x * blockDim.x;

    float sum = 0.0f;

    for (int block = 0; block < gridDim.x; ++block) {
        blockA[localRow * BLOCK_SIZE + localCol] = a[globalRow * n + block * BLOCK_SIZE + localCol];
        blockB[localRow * BLOCK_SIZE + localCol] = b[(block * BLOCK_SIZE + localRow) * n + globalCol];
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += blockA[localRow * BLOCK_SIZE + k] * blockB[k * BLOCK_SIZE + localCol];
        }
        __syncthreads();
    }
    c[globalRow * n + globalCol] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    int size = a.size();
    int bytesSize = n * n * sizeof(float);

    float* gpuBufferA = nullptr;
    float* gpuBufferB = nullptr;
    float* gpuBufferC = nullptr;

    cudaMalloc(&gpuBufferA, bytesSize);
    cudaMalloc(&gpuBufferB, bytesSize);
    cudaMalloc(&gpuBufferC, bytesSize);

    cudaMemcpy(gpuBufferA, a.data(), bytesSize, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuBufferB, b.data(), bytesSize, cudaMemcpyHostToDevice);
    cudaMemset(gpuBufferC, 0, bytesSize);

    int blocks = n / BLOCK_SIZE;
    dim3 threadsDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksDim(blocks, blocks);

    BlockGemmCUDAImpl<<<blocksDim, threadsDim>>>(gpuBufferA, gpuBufferB, gpuBufferC, n);

    std::vector<float> c(size);
    float* cData = c.data();

    cudaDeviceSynchronize();
    cudaMemcpy(cData, gpuBufferC, bytesSize, cudaMemcpyDeviceToHost);
    cudaFree(gpuBufferA);
    cudaFree(gpuBufferB);
    cudaFree(gpuBufferC);

    return c;
}
