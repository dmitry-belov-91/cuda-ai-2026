#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <cuda/std/cmath>

__global__ void GeluKernel(float* devicePtr, int size) {
    constexpr float const1 = 0.044715f;
    constexpr float const2 = 1.59576912f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = devicePtr[idx];
        devicePtr[idx] = val - val / (cuda::std::expf(const2 * val * (1.f + const1 * val * val)) + 1.f);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const int size = input.size();
    const int bSize = size * sizeof(float);

    const float* inPtrHost = input.data();
    float* outPtrHost = nullptr;
    float* devicePtr = nullptr;

    cudaMalloc(&devicePtr, bSize);
    cudaMemcpy(devicePtr, inPtrHost, bSize, cudaMemcpyHostToDevice);

    constexpr int nThreads = 256;
    int nBlocks = (size + nThreads - 1) / nThreads;
    GeluKernel<<<nBlocks, nThreads>>>(devicePtr, size);

    std::vector<float> output(size);
    outPtrHost = output.data();

    cudaDeviceSynchronize();
    cudaMemcpy(outPtrHost, devicePtr, bSize, cudaMemcpyDeviceToHost);
    cudaFree(devicePtr);

    return output;
}
