#include "naive_gemm_cuda.h"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iostream>

__global__ void multKernel(const float * __restrict__ A,
                             const float * __restrict__ B,
                             float * __restrict__ C,
                             int N)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < N)
    {
        float res = 0.;
        for(int k = 0; k < N; ++k)
        {
            res += A[row*N+k]*B[k*N+col];
        }
        C[row*N+col] = res;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    const int tileSize = 16;
    const size_t bytes = n * n * sizeof(float);

    const float* in_a = a.data();
    const float* in_b = b.data();

    std::vector<float> output(n*n);
    float *out_c = output.data();

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, in_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, in_b, bytes, cudaMemcpyHostToDevice);

    int size = (n+tileSize-1)/tileSize;

    dim3 dimBlock(tileSize, tileSize);
    dim3 dimGrid(size, size);

    multKernel <<<dimGrid, dimBlock>>> (d_a, d_b, d_c, n);

    cudaMemcpy(out_c, d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return output;
}
