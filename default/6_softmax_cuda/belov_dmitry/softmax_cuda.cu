#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <cuda/cmath>

__global__ void softMaxFunc(const float* input, float* output, int rowCount, int colCount, int numEl)
{
    extern __shared__ float sharedMem[];
    float* sharedMaxSum = sharedMem; 
    float* sharedExp  = &sharedMem[rowCount];

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < rowCount)
    {
        const int elStart = index*colCount;
        sharedMaxSum[index] = -INFINITY;
        for (int iCol = 0; iCol < colCount; ++iCol)
            if (sharedMaxSum[index] < input[elStart + iCol])
                sharedMaxSum[index] = input[elStart + iCol];
    }

    __syncthreads();
    
    if (index < numEl)
    {
        sharedExp[index] = cuda::std::expf(input[index] - sharedMaxSum[index/colCount]);
    }

    __syncthreads();

    if (index < rowCount)
    {
        const int elStart = index*colCount;
        sharedMaxSum[index] = 0;
        for (int iCol = 0; iCol < colCount; ++iCol)
            sharedMaxSum[index] += sharedExp[elStart + iCol];
    }

    __syncthreads();


    if (index < numEl)
    {
        output[index] = sharedExp[index] / sharedMaxSum[index/colCount];
    }

}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int rowCount) 
{
    const int mtxNumEl = static_cast<int>(input.size());
    const size_t bitMtxNumEl = mtxNumEl * sizeof(float);
    const int colCount = mtxNumEl / rowCount;

    float *deviceInput = nullptr;
    cudaMalloc(&deviceInput, bitMtxNumEl);
    cudaMemcpy(deviceInput, input.data(), bitMtxNumEl, cudaMemcpyHostToDevice);

    float *deviceOutput = nullptr;
    cudaMalloc(&deviceOutput, bitMtxNumEl);

    const int numThreads = rowCount;
    int numBlocks = (static_cast<int>(mtxNumEl) + numThreads - 1) / numThreads;
    size_t sharedMemSize = (rowCount + mtxNumEl) * sizeof(float);
    softMaxFunc<<<numBlocks, numThreads, sharedMemSize>>>(deviceInput, deviceOutput, rowCount, colCount, mtxNumEl);
    
    cudaDeviceSynchronize();

    std::vector<float> output(mtxNumEl);
    cudaMemcpy(output.data(), deviceOutput, bitMtxNumEl, cudaMemcpyDeviceToHost);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    return output;

}