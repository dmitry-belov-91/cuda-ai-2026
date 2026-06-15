#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <vector>
#include <algorithm>
#include <iostream>

#define WARP_SIZE 32
#define WMMA_SIZE 16

using dataT = __half;

__global__ void GEMMv4(const __half* __restrict__ a, 
                       const __half* __restrict__ b, 
                       float* __restrict__ c, 
                       int N) {
    
    int warp_i = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warp_j = blockIdx.y * blockDim.y + threadIdx.y;
    
    int c_row = warp_i * WMMA_SIZE;
    int c_col = warp_j * WMMA_SIZE;
    
    if (c_row >= N || c_col >= N) return;
    
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, 
                           __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, 
                           __half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, 
                           float> acc_frag;
    
    nvcuda::wmma::fill_fragment(acc_frag, 0.0f);
    
    for (int k = 0; k < N; k += WMMA_SIZE) {
        if (c_row + WMMA_SIZE <= N && k + WMMA_SIZE <= N && c_col + WMMA_SIZE <= N) {
            nvcuda::wmma::load_matrix_sync(a_frag, a + c_row * N + k, N);
            nvcuda::wmma::load_matrix_sync(b_frag, b + k * N + c_col, N);
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    
    if (c_row + WMMA_SIZE <= N && c_col + WMMA_SIZE <= N) {
        nvcuda::wmma::store_matrix_sync(c + c_row * N + c_col, acc_frag, N, 
                                        nvcuda::wmma::mem_row_major);
    }
}


__global__ void convertFloat2Half(const float* __restrict__ input, 
                                   __half* __restrict__ output, 
                                   int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

__global__ void kernelGemmSquare(const float* __restrict__ a, 
                                 const float* __restrict__ b, 
                                 float* __restrict__ c, 
                                 int n) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < n && j < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[i * n + k] * b[k * n + j];
        }
        c[i * n + j] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    
    const int lenVec = n * n;
    const int sizeFloat = lenVec * sizeof(float);
    const int sizeHalf = lenVec * sizeof(__half);
    
    float *a_float_dev = nullptr, *b_float_dev = nullptr;
    __half *a_half_dev = nullptr, *b_half_dev = nullptr;
    float *c_dev = nullptr;
    
    cudaMalloc(&a_float_dev, sizeFloat);
    cudaMalloc(&b_float_dev, sizeFloat);
    cudaMalloc(&a_half_dev, sizeHalf);
    cudaMalloc(&b_half_dev, sizeHalf);
    cudaMalloc(&c_dev, sizeFloat);
    
    cudaMemcpy(a_float_dev, a.data(), sizeFloat, cudaMemcpyHostToDevice);
    cudaMemcpy(b_float_dev, b.data(), sizeFloat, cudaMemcpyHostToDevice);

    constexpr int blockSize = 256;
    int gridSize = (lenVec + blockSize - 1) / blockSize;
    convertFloat2Half<<<gridSize, blockSize>>>(a_float_dev, a_half_dev, lenVec);
    convertFloat2Half<<<gridSize, blockSize>>>(b_float_dev, b_half_dev, lenVec);
    cudaDeviceSynchronize();
    
    dim3 block_size(4 * WARP_SIZE, 1);
    dim3 block_count(
        (n + WMMA_SIZE - 1) / WMMA_SIZE,
        (n + WMMA_SIZE - 1) / WMMA_SIZE
    );
    
    GEMMv4<<<block_count, block_size>>>(a_half_dev, b_half_dev, c_dev, n);
    
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
    
    std::vector<float> c(lenVec);
    cudaMemcpy(c.data(), c_dev, sizeFloat, cudaMemcpyDeviceToHost);
    
    cudaFree(a_float_dev);
    cudaFree(b_float_dev);
    cudaFree(a_half_dev);
    cudaFree(b_half_dev);
    cudaFree(c_dev);
    
    return c;
}
