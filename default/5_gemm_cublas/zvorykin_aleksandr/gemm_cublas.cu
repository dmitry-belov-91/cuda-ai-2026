#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda/cmath>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n)
{
    // Place your implementation here
    size_t size = n * n;
    size_t dataSize = size * sizeof(float);
    std::vector<float> c(size);

    float *deviceA = nullptr;
    cudaMalloc(&deviceA, dataSize);
    float *deviceB = nullptr;
    cudaMalloc(&deviceB, dataSize);
    float *deviceC = nullptr;
    cudaMalloc(&deviceC, dataSize);
    
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    cudaMemcpy(deviceA, a.data(), dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, b.data(), dataSize, cudaMemcpyHostToDevice);

    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, deviceB, n, deviceA, n, &beta, deviceC, n);
   
    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), deviceC, dataSize, cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    cublasDestroy(cublasHandle); 

    return c;
}
