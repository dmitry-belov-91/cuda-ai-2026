#include "naive_gemm_cuda.h"

#include <cuda/cmath>
#include <cuda_runtime.h>

__global__ void gemm_kernel(const float* in_a, float* in_b, float* out, size_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // todo cache data

    if (idx < n && idy < n) {
        int i = idx;
        float * c_i_row = out + i * n;
        for (size_t k = 0; k < n; ++k) {
            const float * b_k_row = in_b + k * n;
            float a_i_k = in_a[i * n + k];
            int j = idy;
            c_i_row[j] += a_i_k * b_k_row[j];
        }
    }
}


std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);


    const size_t memsize = a.size() * sizeof(float) ;
    float *in_a = nullptr;
    float *in_b = nullptr;
    float *out = nullptr;

    cudaMalloc((void**)&in_a, memsize);
    cudaMalloc((void**)&in_b, memsize);
    cudaMalloc((void**)&out, memsize);

    // Straightforward approach
    cudaMemcpy(in_a, a.data(), memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(in_b, b.data(), memsize, cudaMemcpyHostToDevice);

    gemm_kernel<<<numBlocks, threadsPerBlock>>>(in_a, in_b, out, n);

    std::vector<float> result(a.size());

    cudaMemcpy(result.data(), out, memsize, cudaMemcpyDeviceToHost); 

    cudaFree(in_a);
    cudaFree(in_b);
    cudaFree(out);

    return result;
}