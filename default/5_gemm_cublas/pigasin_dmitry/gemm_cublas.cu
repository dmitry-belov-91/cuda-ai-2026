#include "gemm_cublas.h"

#include <cublas_v2.h>

#include <vector>
#include <thread>


std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {
    const size_t numElem = a.size();

    std::vector<float> c;
    std::thread t([&](){c.resize(numElem);});

    float *gpuA, *gpuB, *gpuC;
    const size_t numBytes = numElem * sizeof(float);
    cudaMalloc(&gpuA, numBytes);
    cudaMalloc(&gpuB, numBytes);
    cudaMalloc(&gpuC, numBytes);

    cudaMemcpy(gpuA, a.data(), numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuB, b.data(), numBytes, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, gpuB, n, gpuA, n, &beta, gpuC, n);
    cublasDestroy(handle);

    t.join();
    cudaMemcpy(c.data(), gpuC, numBytes, cudaMemcpyDeviceToHost);

    cudaFree(gpuA);
    cudaFree(gpuB);
    cudaFree(gpuC);

    return c;
}
