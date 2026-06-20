#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

struct Context {
    cublasHandle_t handle{};

    float* vidA = nullptr;
    float* vidB = nullptr;
    float* vidC = nullptr;

    size_t allocated_bytes = 0;

    Context() {
        cublasCreate(&handle);
    }

    void EnsureCapacity(size_t bytes) {
        if (bytes <= allocated_bytes)
            return;

        if (vidA) cudaFree(vidA);
        if (vidB) cudaFree(vidB);
        if (vidC) cudaFree(vidC);

        cudaMalloc(&vidA, bytes);
        cudaMalloc(&vidB, bytes);
        cudaMalloc(&vidC, bytes);

        allocated_bytes = bytes;
    }

    ~Context() {
        if (vidA) cudaFree(vidA);
        if (vidB) cudaFree(vidB);
        if (vidC) cudaFree(vidC);

        cublasDestroy(handle);
    }
};

static Context ctx;


std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {

    const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    ctx.EnsureCapacity(bytes);

    cudaMemcpy(ctx.vidA, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx.vidB, b.data(), bytes, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n, n, n,
        &alpha,
        ctx.vidB, n,
        ctx.vidA, n,
        &beta,
        ctx.vidC, n);

    std::vector<float> c(n * n);

    cudaMemcpy(c.data(), ctx.vidC, bytes, cudaMemcpyDeviceToHost);

    return c;
}