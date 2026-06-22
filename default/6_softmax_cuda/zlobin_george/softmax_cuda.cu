#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <cfloat>
#include <vector>

constexpr int BLOCK_SIZE = 256;

__global__ void SoftmaxCUDAKernelImpl(const float* input,
                                      float* output,
                                      int rowSize) {
    __shared__ float reductionBuffer[BLOCK_SIZE];

    const int row = blockIdx.x;
    const int threadX = threadIdx.x;
    const int offset = row * rowSize;

    float rowMax = -FLT_MAX;
    for (int i = threadX; i < rowSize; i += BLOCK_SIZE) {
        rowMax = fmaxf(rowMax, input[offset + i]);
    }

    reductionBuffer[threadX] = rowMax;
    __syncthreads();
    for (int range = BLOCK_SIZE / 2; range > 0; range >>= 1) {
        if (threadX < range) {
            reductionBuffer[threadX] = fmaxf(reductionBuffer[threadX], reductionBuffer[threadX + range]);
        }
        __syncthreads();
    }
    rowMax = reductionBuffer[0];

    float rowSum = 0.f;
    for (int i = threadX; i < rowSize; i += BLOCK_SIZE) {
        output[offset + i] = __expf(input[offset + i] - rowMax);
        rowSum += output[offset + i];
    }

    reductionBuffer[threadX] = rowSum;
    __syncthreads();
    for (int range = BLOCK_SIZE / 2; range > 0; range >>= 1) {
        if (threadX < range) {
            reductionBuffer[threadX] += reductionBuffer[threadX + range];
        }
        __syncthreads();
    }
    float invSum = 1.0f / reductionBuffer[0];

    for (int i = threadX; i < rowSize; i += BLOCK_SIZE) {
        output[offset + i] *= invSum;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input,
                               int row_count) {
    const int size = input.size();
    const int bSize = size * sizeof(float);
    const int rowSize = size / row_count;

    float* outHostPtr = nullptr;
    float* inDevicePtr = nullptr;
    float* outDevicePtr = nullptr;

    cudaMalloc(&inDevicePtr, bSize);
    cudaMalloc(&outDevicePtr, bSize);

    cudaMemcpy(inDevicePtr, input.data(), bSize, cudaMemcpyHostToDevice);

    SoftmaxCUDAKernelImpl<<<row_count, BLOCK_SIZE>>>(inDevicePtr, outDevicePtr, rowSize);

    std::vector<float> output(size);
    outHostPtr = output.data();

    cudaDeviceSynchronize();
    cudaMemcpy(outHostPtr, outDevicePtr, bSize, cudaMemcpyDeviceToHost);
    cudaFree(inDevicePtr);
    cudaFree(outDevicePtr);

    return output;
}
