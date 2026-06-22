#include "gemm_cublas.h"

#include <vector>
#include <algorithm>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    const int size = a.size();
    const int bSize = size * sizeof(float);
    
    float* cHostPtr = nullptr;
    float* aDevicePtr = nullptr;
    float* bDevicePtr = nullptr;
    float* cDevicePtr = nullptr;
    
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMalloc(&aDevicePtr, bSize);
    cudaMalloc(&bDevicePtr, bSize);
    cudaMalloc(&cDevicePtr, bSize);

    cudaMemcpy(aDevicePtr, a.data(), bSize, cudaMemcpyHostToDevice);
    cudaMemcpy(bDevicePtr, b.data(), bSize, cudaMemcpyHostToDevice);

    float alpha = 1.f;
    float beta = 0.f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, bDevicePtr, n, aDevicePtr, n, &beta, cDevicePtr, n);

    std::vector<float> c(size);
    cHostPtr = c.data();
    
    cudaMemcpy(cHostPtr, cDevicePtr, bSize, cudaMemcpyDeviceToHost);
    cudaFree(aDevicePtr);
    cudaFree(bDevicePtr);
    cudaFree(cDevicePtr);

    return c;
}
