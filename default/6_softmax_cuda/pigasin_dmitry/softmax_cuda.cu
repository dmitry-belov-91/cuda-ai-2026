#include "softmax_cuda.h"

#include <thread>

class Ctx {
public:
    float* matGpu = nullptr;
    float* colGpu = nullptr;

    void prepareMem(const size_t matNumBytes, const size_t colNumBytes) {
        if (matNumBytes > this->matNumBytes) {
            if (matGpu)
                cudaFree(matGpu);
            cudaMalloc(&matGpu, matNumBytes);
            this->matNumBytes = matNumBytes;
        }

        if (colNumBytes > this->colNumBytes) {
            if (colGpu)
                cudaFree(colGpu);
            cudaMalloc(&colGpu, colNumBytes);
            this->colNumBytes = colNumBytes;
        }
    }

    ~Ctx() {
        cudaFree(matGpu);
        cudaFree(colGpu);
    }

private:
    size_t matNumBytes = 0;
    size_t colNumBytes = 0;
};

static Ctx ctx;


__global__ void getRowMaxes(float* mat, size_t m, size_t n, float* rowMaxes) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= m) {
        return;
    }

    float max = -INFINITY;
    float x = 0.f;
    for (size_t j = 0; j < n; ++j) {
        x = mat[i * n + j];
        if (x > max) {
            max = x;
        }
    }
    rowMaxes[i] = max;
}

__global__ void exp(float* mat, size_t m, size_t n, float* rowMaxes) {
    const size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= m || j >= n) {
        return;
    }

    mat[i * n + j] = std::exp(mat[i * n + j] - rowMaxes[i]);
}

__global__ void getRowSums(float* mat, size_t m, size_t n, float* rowSums) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= m) {
        return;
    }

    float sum = 0.f;
    for (size_t j = 0; j < n; ++j) {
        sum += mat[i * n + j];
    }
    rowSums[i] = sum;
}

__global__ void inv(float* x, size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    x[i] = 1.f / x[i];
}

__global__ void mul(float* mat, size_t m, size_t n, float* rowMuls) {
    const size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= m || j >= n) {
        return;
    }

    mat[i * n + j] *= rowMuls[i];
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& mat, int m) {
    std::vector<float> out;
    const size_t numElem = mat.size();
    std::thread t([&](){ out.resize(numElem); });

    const size_t n = numElem / m;
    const size_t matNumBytes = numElem * sizeof(float);
    const size_t colNumBytes = m * sizeof(float);
    ctx.prepareMem(matNumBytes, colNumBytes);
    
    cudaMemcpy(ctx.matGpu, mat.data(), matNumBytes, cudaMemcpyHostToDevice);

    const int blockSize1d = 256;
    const int numBlocks1d = (m + blockSize1d - 1) / blockSize1d;
    getRowMaxes<<<numBlocks1d, blockSize1d>>>(ctx.matGpu, m, n, ctx.colGpu);

    dim3 blockSize2d(16, 16);
    dim3 numBlocks2d((n + blockSize2d.x - 1) / blockSize2d.x, (m + blockSize2d.y - 1) / blockSize2d.y);
    exp<<<numBlocks2d, blockSize2d>>>(ctx.matGpu, m, n, ctx.colGpu);

    getRowSums<<<numBlocks1d, blockSize1d>>>(ctx.matGpu, m, n, ctx.colGpu);

    inv<<<numBlocks1d, blockSize1d>>>(ctx.colGpu, m);

    mul<<<numBlocks2d, blockSize2d>>>(ctx.matGpu, m, n, ctx.colGpu);

    t.join();
    cudaMemcpy(out.data(), ctx.matGpu, matNumBytes, cudaMemcpyDeviceToHost);

    return out;
}
