import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time

_kernel_source = """
__global__ void layernorm_kernel(const float* input, const float* gamma, const float* beta,
                                 float* output, int row_size, float eps) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    const float* row_in = input + row * row_size;
    float* row_out = output + row * row_size;

    float sum_x = 0.0f;
    for (int i = tid; i < row_size; i += stride)
        sum_x += row_in[i];
    sdata[tid] = sum_x;
    __syncthreads();

    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / row_size;

    float sum_sq = 0.0f;
    for (int i = tid; i < row_size; i += stride) {
        float diff = row_in[i] - mean;
        sum_sq += diff * diff;
    }
    sdata[tid] = sum_sq;
    __syncthreads();

    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float var = sdata[0] / row_size + eps;
    float inv_std = rsqrtf(var);

    for (int i = tid; i < row_size; i += stride)
        row_out[i] = gamma[i] * (row_in[i] - mean) * inv_std + beta[i];
}
"""

_mod = None

def _get_module():
    global _mod
    if _mod is None:
        _mod = SourceModule(_kernel_source, arch='sm_86')
    return _mod

def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    input = np.asarray(input, dtype=np.float32).ravel()
    gamma = np.asarray(gamma, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)

    n_total = input.size
    n_rows = n_total // row_size
    assert n_total % row_size == 0, "Input length must be divisible by row_size"

    threads = min(row_size, 256) if row_size > 256 else row_size

    d_input = cuda.mem_alloc(input.nbytes)
    d_gamma = cuda.mem_alloc(gamma.nbytes)
    d_beta = cuda.mem_alloc(beta.nbytes)
    d_output = cuda.mem_alloc(input.nbytes)

    cuda.memcpy_htod(d_input, input)
    cuda.memcpy_htod(d_gamma, gamma)
    cuda.memcpy_htod(d_beta, beta)

    func = _get_module().get_function("layernorm_kernel")
    func(d_input, d_gamma, d_beta, d_output,
         np.int32(row_size), np.float32(eps),
         block=(threads, 1, 1), grid=(n_rows, 1),
         shared=threads * 4)

    output = np.empty(n_total, dtype=np.float32)
    cuda.memcpy_dtoh(output, d_output)

    return output
