#include "gelu_cuda.h"

#include <cuda_runtime_api.h>
#include <cuda/cmath>
#include <memory>
#include <vector>

__global__ void geluKernel(float * data, int size)
{
    const float coef0 = -1.595769f;
    const float coef1 = 0.044715f;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
    {
        const float x = data[idx];
        const float expVal = cuda::std::expf(coef0 * x * (1.0f + coef1 * x * x));
        data[idx] = x / (expVal + 1);
    }
}

std::vector<float> GeluCUDA(const std::vector<float> &input)
{
    const size_t vecLen = input.size();
    const size_t size = vecLen * sizeof(float);

    float * devInput = nullptr;

    cudaMalloc(&devInput, size);
    cudaMemcpy(devInput, input.data(), size, cudaMemcpyHostToDevice);

    constexpr int numThreads = 256;
    int numBlocks = cuda::ceil_div(vecLen, numThreads);

    geluKernel<<<numBlocks, numThreads>>>(devInput, static_cast<int>(size));

    std::vector<float> output(vecLen);
    cudaDeviceSynchronize();
    cudaMemcpy(output.data(), devInput, size, cudaMemcpyDeviceToHost);

    cudaFree(devInput);

    return output;
}