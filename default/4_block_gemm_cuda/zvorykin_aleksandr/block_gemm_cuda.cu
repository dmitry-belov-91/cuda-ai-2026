#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstring>
#include <cstdlib>

// CUDA constants
constexpr int s_BlockM = 64;
constexpr int s_BlockN = 64;
constexpr int s_BlockK = 32;
constexpr int s_TileM   = 4;
constexpr int s_TileN   = 8;
constexpr int s_Pad     = 1;

// Anonymous namespace
namespace
{
  __global__ void optBlockGemmImpl(const float* __restrict__ a,
                                   const float* __restrict__ b,
                                   float* __restrict__ c,
                                   int n)
    {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int warpRow = (blockIdx.y * blockDim.y + threadIdx.y) / warpSize;
        const int warpCol = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
        const int rowBase = warpRow * s_TileM * 8;
        const int colBase = warpCol * s_TileN * 8;
        
        __shared__ float as0[s_BlockK][s_BlockM + s_Pad];
        __shared__ float as1[s_BlockK][s_BlockM + s_Pad];
        __shared__ float bs0[s_BlockK][s_BlockN + s_Pad];
        __shared__ float bs1[s_BlockK][s_BlockN + s_Pad];
        
        float (*asCur)[s_BlockM + s_Pad] = as0;
        float (*asNxt)[s_BlockM + s_Pad] = as1;
        float (*bsCur)[s_BlockN + s_Pad] = bs0;
        float (*bsNxt)[s_BlockN + s_Pad] = bs1;
        
        float acc[s_TileM * s_TileN];
        #pragma unroll
        for (int i = 0; i < s_TileM * s_TileN; ++i)
          acc[i] = 0.0f;
        
        const int aLoadRow = ty * 8;
        const int aLoadCol = 0;
        const int bLoadRow = 0;
        const int bLoadCol = tx * 8;
        
        #pragma unroll
        for (int i = 0; i < s_BlockK; ++i)
        {
            const float* aPtr = a + (blockIdx.y * s_BlockM + aLoadRow) * n + i;
            float4 aVals = *reinterpret_cast<const float4*>(aPtr);
            asCur[i][aLoadRow + 0] = aVals.x;
            asCur[i][aLoadRow + 1] = aVals.y;
            asCur[i][aLoadRow + 2] = aVals.z;
            asCur[i][aLoadRow + 3] = aVals.w;
        
            const float* bPtr = b + i * n + (blockIdx.x * s_BlockN + bLoadCol);
            float4 bVals = *reinterpret_cast<const float4*>(bPtr);
            bsCur[i][bLoadCol + 0] = bVals.x;
            bsCur[i][bLoadCol + 1] = bVals.y;
            bsCur[i][bLoadCol + 2] = bVals.z;
            bsCur[i][bLoadCol + 3] = bVals.w;
        }
        __syncthreads();
        
        for (int bk = s_BlockK; bk < n; bk += s_BlockK) 
        {
            #pragma unroll
            for (int i = 0; i < s_BlockK; ++i) 
            {
                const float* aPtr = a + (blockIdx.y * s_BlockM + aLoadRow) * n + bk + i;
                float4 aVals = *reinterpret_cast<const float4*>(aPtr);
                asNxt[i][aLoadRow + 0] = aVals.x;
                asNxt[i][aLoadRow + 1] = aVals.y;
                asNxt[i][aLoadRow + 2] = aVals.z;
                asNxt[i][aLoadRow + 3] = aVals.w;
        
                const float* bPtr = b + (bk + i) * n + (blockIdx.x * s_BlockN + bLoadCol);
                float4 bVals = *reinterpret_cast<const float4*>(bPtr);
                bsNxt[i][bLoadCol + 0] = bVals.x;
                bsNxt[i][bLoadCol + 1] = bVals.y;
                bsNxt[i][bLoadCol + 2] = bVals.z;
                bsNxt[i][bLoadCol + 3] = bVals.w;
            }
            __syncthreads();
        
            #pragma unroll
            for (int k = 0; k < s_BlockK; ++k)
            {
                float aReg[s_TileM];
                float bReg[s_TileN];
        
                #pragma unroll
                for (int i = 0; i < s_TileM; ++i)
                    aReg[i] = asCur[k][ty * s_TileM * 8 + i * 8 + tx];
        
                #pragma unroll
                for (int j = 0; j < s_TileN; ++j)
                    bReg[j] = bsCur[k][tx * s_TileN * 8 + j * 8 + ty];
        
                #pragma unroll
                for (int i = 0; i < s_TileM; ++i) 
                {
                    #pragma unroll
                    for (int j = 0; j < s_TileN; ++j) 
                    {
                        acc[i * s_TileN + j] += aReg[i] * bReg[j];
                    }
                }
            }
            __syncthreads();
        
            float (*tmpA)[s_BlockM + s_Pad] = asCur;
            asCur = asNxt;
            asNxt = tmpA;
            
            float (*tmpB)[s_BlockN + s_Pad] = bsCur;
            bsCur = bsNxt;
            bsNxt = tmpB;
        }
        
        #pragma unroll
        for (int k = 0; k < s_BlockK; ++k)
        {
            float aReg[s_TileM];
        }
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    // Place your implementation here

    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    float *pinnedA;
    cudaHostAlloc(&pinnedA, bytes, cudaHostAllocMapped);
    float *pinnedB;
    cudaHostAlloc(&pinnedB, bytes, cudaHostAllocMapped);
    float *pinnedC;
    cudaHostAlloc(&pinnedC, bytes, cudaHostAllocMapped);

    std::memcpy(pinnedA, a.data(), bytes);
    std::memcpy(pinnedB, b.data(), bytes);
    std::memset(pinnedC, 0, bytes);

    float *deviceA;
    cudaMalloc(&deviceA, bytes);
    float *deviceB;
    cudaMalloc(&deviceB, bytes);
    float *deviceC;
    cudaMalloc(&deviceC, bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(deviceA, pinnedA, bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(deviceB, pinnedB, bytes, cudaMemcpyHostToDevice, stream);

    dim3 blockDim(s_TileN, s_TileM);
    dim3 gridDim(n / s_BlockN, n / s_BlockM);
    optBlockGemmImpl<<<gridDim, blockDim, 0, stream>>>(deviceA, deviceB, deviceC, n);

    cudaMemcpyAsync(pinnedC, deviceC, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    std::vector<float> c(n * n);
    std::memcpy(c.data(), pinnedC, bytes);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    cudaFreeHost(pinnedA);
    cudaFreeHost(pinnedB);
    cudaFreeHost(pinnedC);

    return c;
}
