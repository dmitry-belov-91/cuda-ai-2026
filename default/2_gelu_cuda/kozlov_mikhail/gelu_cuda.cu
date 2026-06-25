#include "gelu_cuda.h"
#include <cuda_runtime.h>

__global__ void gelu_kernel(const float* __restrict__ src, float* __restrict__ dst, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        float x = src[idx];
        float inner = 0.79788456f * x * (1.0f + 0.044715f * x * x);
        dst[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = input.size();
    if (n == 0) return std::vector<float>();

    static cudaStream_t stream = nullptr;
    static float* dev_src = nullptr;
    static float* dev_dst = nullptr;
    static int previous_size = 0;

    if (stream == nullptr) {
        cudaStreamCreate(&stream);
    }

    if (n > previous_size) {
        if (dev_src) cudaFree(dev_src);
        if (dev_dst) cudaFree(dev_dst);
        cudaMalloc(&dev_src, n * sizeof(float));
        cudaMalloc(&dev_dst, n * sizeof(float));
        previous_size = n;
    }

    cudaMemcpyAsync(dev_src, input.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_kernel<<<blocks, threads, 0, stream>>>(dev_src, dev_dst, n);

    std::vector<float> result(n);
    cudaMemcpyAsync(result.data(), dev_dst, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return result;
}
