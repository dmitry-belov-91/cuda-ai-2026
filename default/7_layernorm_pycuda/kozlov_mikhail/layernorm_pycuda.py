import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

_kernel_code = SourceModule(r"""
#include <math.h>

#define BLOCK_SIZE 256

__global__ void layer_norm_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int row_size,
    float eps)
{
    __shared__ float sdata[BLOCK_SIZE];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int base = row * row_size;

    // Compute mean
    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += BLOCK_SIZE) {
        local_sum += input[base + i];
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    float mean = sdata[0] / row_size;
    __syncthreads();

    // Compute variance
    float local_var = 0.0f;
    for (int i = tid; i < row_size; i += BLOCK_SIZE) {
        float diff = input[base + i] - mean;
        local_var += diff * diff;
    }

    sdata[tid] = local_var;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    float inv_std = rsqrtf((sdata[0] / row_size) + eps);

    // Apply layer normalization
    for (int i = tid; i < row_size; i += BLOCK_SIZE) {
        float x = input[base + i];
        output[base + i] = (x - mean) * inv_std * gamma[i] + beta[i];
    }
}
""", options=["-O3", "-use_fast_math"])

_layer_norm_kernel = _kernel_code.get_function("layer_norm_kernel")


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    input = np.asarray(input, dtype=np.float32)
    gamma = np.asarray(gamma, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)

    row_count = input.size // row_size
    output = np.empty_like(input)

    d_input = cuda.mem_alloc(input.nbytes)
    d_gamma = cuda.mem_alloc(gamma.nbytes)
    d_beta = cuda.mem_alloc(beta.nbytes)
    d_output = cuda.mem_alloc(output.nbytes)

    cuda.memcpy_htod(d_input, input)
    cuda.memcpy_htod(d_gamma, gamma)
    cuda.memcpy_htod(d_beta, beta)

    _layer_norm_kernel(
        d_input,
        d_gamma,
        d_beta,
        d_output,
        np.int32(row_size),
        np.float32(eps),
        block=(256, 1, 1),
        grid=(int(row_count), 1, 1),
    )

    cuda.memcpy_dtoh(output, d_output)

    d_input.free()
    d_gamma.free()
    d_beta.free()
    d_output.free()

    return output
