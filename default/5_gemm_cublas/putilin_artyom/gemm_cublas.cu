#include "gemm_cublas.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <cublas_v2.h>
#include <algorithm>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n)
{
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    const size_t N = n * n;
    const size_t bytes = N * sizeof(float);

    float *d_A;
    float *d_B;
    float *d_C;

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    std::vector<float> c(N);

    cudaMemcpy(d_A, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), bytes, cudaMemcpyHostToDevice);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n, &alpha, d_B, n, d_A, n, &beta, d_C, n);
    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return c;
}
