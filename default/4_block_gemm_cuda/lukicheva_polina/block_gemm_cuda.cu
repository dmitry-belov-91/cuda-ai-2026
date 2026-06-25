#include <cmath>
#include <iostream>
#include <random>
#include <chrono>

#include "block_gemm_cuda.h"

#define BLOCK_SIZE 16

__global__ void BlockGemmKernel(float *a, float *b, float *c, int n)
{
    int localRow = threadIdx.y;
    int localCol = threadIdx.x;
    int globalRow = threadIdx.y + blockIdx.y * blockDim.y;
    int globalCol = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float aBlock[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float bBlock[BLOCK_SIZE * BLOCK_SIZE];

    float sum = 0;

    for (int blIdx = 0; blIdx < gridDim.x; ++blIdx)
    {
        aBlock[localRow * BLOCK_SIZE + localCol] = a[globalRow * n + blIdx * BLOCK_SIZE + localCol];
        bBlock[localRow * BLOCK_SIZE + localCol] = b[(blIdx * BLOCK_SIZE + localRow) * n + globalCol];
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            sum += aBlock[localRow * BLOCK_SIZE + k] * bBlock[k * BLOCK_SIZE + localCol];
        }
        __syncthreads();
    }
    c[globalRow * n + globalCol] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b,
                                 int n)
{
    int nElems = n * n;
    std::vector<float> c(nElems, 0);

    const float *aHost = a.data();
    const float *bHost = b.data();
    float *cHost = c.data();

    float *aDevice = nullptr;
    float *bDevice = nullptr;
    float *cDevice = nullptr;
    cudaMalloc(&aDevice, nElems * sizeof(float));
    cudaMalloc(&bDevice, nElems * sizeof(float));
    cudaMalloc(&cDevice, nElems * sizeof(float));

    cudaMemcpy(aDevice, aHost, nElems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bDevice, bHost, nElems * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 num_blocks(n / BLOCK_SIZE, n / BLOCK_SIZE);
    BlockGemmKernel<<<num_blocks, threadsPerBlock>>>(aDevice, bDevice, cDevice, n);

    cudaMemcpy(cHost, cDevice, nElems * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(aDevice);
    cudaFree(bDevice);
    cudaFree(cDevice);

    return c;
}

#if 0
std::vector<float> NaiveGemmScalar(const std::vector<float> &a,
                                   const std::vector<float> &b,
                                   int n)
{
    std::vector<float> c(n * n, 0);

    const float *aPtr = a.data();
    const float *bPtr = b.data();
    float *cPtr = c.data();

#pragma omp parallel for
    for (int i = 0; i < n; ++i)
    {
        for (int k = 0; k < n; ++k)
        {
            float a_ik = aPtr[i * n + k];

            for (int j = 0; j < n; ++j)
            {
                c[i * n + j] += a_ik * b[k * n + j];
            }
        }
    }

    return c;
}

int main()
{
    size_t n = 2 << 10;
    size_t nElems = n * n;
    std::vector<float> a(nElems), b(nElems);
    for (size_t i = 0; i < nElems; ++i)
    {
        a[i] = ((float)rand() / RAND_MAX) * 20.f - 10.f;
        b[i] = ((float)rand() / RAND_MAX) * 20.f - 10.f;
    }

    auto c_ref = NaiveGemmScalar(a, b, n);
    auto c_cuda = BlockGemmCUDA(a, b, n);

    float error = 0.0f;
    for (size_t i = 0; i < nElems; ++i)
    {
        error = std::max(std::abs(c_ref[i] - c_cuda[i]), error);
    }
    std::cout << "Absolute max error: " << error << std::endl;

    int nIters = 10;
    double min_t = 0.f;

    for (int i = 0; i < nIters; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        c_cuda = BlockGemmCUDA(a, b, n);
        std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;
        double t = duration.count();
        min_t = i == 0 ? t : std::min(min_t, t);
    }

    std::cout << "Min execution time: \t" << min_t << std::endl;
}
#endif