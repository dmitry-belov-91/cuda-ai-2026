#include "naive_gemm_cuda.h"

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda/cmath>

#define THREADS_PER_BLOCK 16

__global__ void naive_matrix_mul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N)
{
     // Calculate the global row and column index for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < N; ++k)
        {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n)
{
    size_t N = n * n;
    size_t mtxSize = N * sizeof(float);
    std::vector<float> c(N);

    float *data_ptr = nullptr;

    cudaMalloc(&data_ptr, 3 * mtxSize);
    cudaMemcpy(data_ptr, a.data(), mtxSize, cudaMemcpyHostToDevice);
    cudaMemcpy(data_ptr + N, b.data(), mtxSize, cudaMemcpyHostToDevice);
    cudaMemset(data_ptr + 2 * N, 0, mtxSize);

    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    naive_matrix_mul_kernel<<<blocksPerGrid, threadsPerBlock>>>(data_ptr, data_ptr + N, data_ptr + 2*N, n);

    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), data_ptr + 2*N, mtxSize, cudaMemcpyDeviceToHost);
    cudaFree(data_ptr);

    return c;

}
