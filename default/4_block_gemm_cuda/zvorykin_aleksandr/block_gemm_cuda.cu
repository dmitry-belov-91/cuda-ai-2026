#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <cstring>
#include <vector>

// CUDA constants
constexpr int s_BlockM = 64;
constexpr int s_BlockN = 64;
constexpr int s_BlockK = 32;
constexpr int s_TileM  = 4;
constexpr int s_TileN  = 8;
constexpr int s_Pad    = 1;

// Anonymous namespace
namespace
{
    __global__ void optBlockGemmImpl(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int n)
    {
        const int blockRow = blockIdx.y * s_BlockM;
        const int blockCol = blockIdx.x * s_BlockN;
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int row0 = blockRow + ty * s_TileM;
        const int col0 = blockCol + tx * s_TileN;

        __shared__ float as0[s_BlockK][s_BlockM + s_Pad];
        __shared__ float as1[s_BlockK][s_BlockM + s_Pad];
        __shared__ float bs0[s_BlockK][s_BlockN + s_Pad];
        __shared__ float bs1[s_BlockK][s_BlockN + s_Pad];

        float (*AsCur)[s_BlockM + s_Pad] = as0;
        float (*AsNxt)[s_BlockM + s_Pad] = as1;
        float (*BsCur)[s_BlockN + s_Pad] = bs0;
        float (*BsNxt)[s_BlockN + s_Pad] = bs1;

        float acc[s_TileM * s_TileN];
#pragma unroll
        for (int i = 0; i < s_TileM * s_TileN; ++i) acc[i] = 0.0f;

        int kLoad = min(s_BlockK, n);
        for (int k = 0; k < kLoad; ++k)
        {
            for (int i = 0; i < s_TileM; ++i)
            {
                int gRow = row0 + i;
                AsCur[k][gRow] = (gRow < n) ? A[gRow * n + k] : 0.0f;
            }
            for (int j = 0; j < s_TileN; ++j)
            {
                int gCol = col0 + j;
                BsCur[k][gCol] = (gCol < n) ? B[k * n + gCol] : 0.0f;
            }
        }
        __syncthreads();

        for (int bk = s_BlockK; bk < n; bk += s_BlockK)
        {
            int Knext = min(s_BlockK, n - bk);
            for (int k = 0; k < Knext; ++k)
            {
                for (int i = 0; i < s_TileM; ++i)
                {
                    int gRow = row0 + i;
                    AsNxt[k][gRow] = (gRow < n) ? A[gRow * n + (bk + k)] : 0.0f;
                }
                for (int j = 0; j < s_TileN; ++j)
                {
                    int gCol = col0 + j;
                    BsNxt[k][gCol] = (gCol < n) ? B[(bk + k) * n + gCol] : 0.0f;
                }
            }
            __syncthreads();

            for (int k = 0; k < kLoad; ++k)
            {
#pragma unroll
                for (int i = 0; i < s_TileM; ++i)
                {
                    float aReg = AsCur[k][row0 + i];
#pragma unroll
                    for (int j = 0; j < s_TileN; ++j)
                    {
                        float bReg = BsCur[k][col0 + j];
                        acc[i * s_TileN + j] += aReg * bReg;
                    }
                }
            }
            __syncthreads();

            float (*tmpA)[s_BlockM + s_Pad] = AsCur; AsCur = AsNxt; AsNxt = tmpA;
            float (*tmpB)[s_BlockN + s_Pad] = BsCur; BsCur = BsNxt; BsNxt = tmpB;
            kLoad = Knext;
        }

        if (kLoad < s_BlockK)
        {
            for (int k = 0; k < kLoad; ++k)
            {
#pragma unroll
                for (int i = 0; i < s_TileM; ++i)
                {
                    float aReg = AsCur[k][row0 + i];
#pragma unroll
                    for (int j = 0; j < s_TileN; ++j)
                    {
                        float bReg = BsCur[k][col0 + j];
                        acc[i * s_TileN + j] += aReg * bReg;
                    }
                }
            }
        }

#pragma unroll
        for (int i = 0; i < s_TileM; ++i)
        {
            int gRow = row0 + i;
            if (gRow >= n)
                continue;
#pragma unroll
            for (int j = 0; j < s_TileN; ++j)
            {
                int gCol = col0 + j;
                if (gCol >= n)
                    continue;
                C[gRow * n + gCol] = acc[i * s_TileN + j];
            }
        }
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    // Place your implementation here
    const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

    float *hA = nullptr, *hB = nullptr, *hC = nullptr;
    cudaHostAlloc(&hA, bytes, cudaHostAllocMapped);
    cudaHostAlloc(&hB, bytes, cudaHostAllocMapped);
    cudaHostAlloc(&hC, bytes, cudaHostAllocMapped);

    std::memcpy(hA, a.data(), bytes);
    std::memcpy(hB, b.data(), bytes);
    std::memset(hC, 0, bytes);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(dA, hA, bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dB, hB, bytes, cudaMemcpyHostToDevice, stream);

    dim3 blockDim(s_TileM, s_TileN);
    dim3 gridDim((n + s_BlockN - 1) / s_BlockN,
                 (n + s_BlockM - 1) / s_BlockM);

    optBlockGemmImpl<<<gridDim, blockDim, 0, stream>>>(dA, dB, dC, n);

    cudaMemcpyAsync(hC, dC, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    std::vector<float> c(n * n);
    std::memcpy(c.data(), hC, bytes);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFreeHost(hA);
    cudaFreeHost(hB);
    cudaFreeHost(hC);

    return c;
}
