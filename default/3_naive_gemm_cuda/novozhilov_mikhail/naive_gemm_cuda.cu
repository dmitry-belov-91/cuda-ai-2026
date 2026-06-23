#include "naive_gemm_cuda.h"

#include <cstdlib>
#include <vector>
#include <algorithm>

__global__ void matmulKernel(const float * __restrict__ A,
                             const float * __restrict__ B,
                             float * __restrict__ C,
                             int N)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        float res = 0.;
        for(int k = 0; k < N; ++k)
        {
            res += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = res;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    const int tileSize = 16;
    const unsigned int count = n * n * sizeof(float);

    const float* in_a = a.data();
    const float* in_b = b.data();

    std::vector<float> output(n * n);
    float *out_c = output.data();

    float *g_a, *g_b, *g_c;
    cudaMalloc(&g_a, count);
    cudaMalloc(&g_b, count);
    cudaMalloc(&g_c, count);

    cudaMemcpy(g_a, in_a, count, cudaMemcpyHostToDevice);
    cudaMemcpy(g_b, in_b, count, cudaMemcpyHostToDevice);

    int size = (n + tileSize - 1) / tileSize;

    dim3 dimBlock(tileSize, tileSize);
    dim3 dimGrid(size, size);

    matmulKernel <<<dimGrid, dimBlock>>> (g_a, g_b, g_c, n);

    cudaMemcpy(out_c, g_c, count, cudaMemcpyDeviceToHost);

    cudaFree(g_a);
    cudaFree(g_b);
    cudaFree(g_c);

    return output;
}
