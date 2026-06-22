import cupy as cp
import numpy as np

layernormImpl = cp.RawKernel(r'''
constexpr int BLOCK_SIZE = 32;

extern "C" __global__
void layernormImpl(const float* input, const float* gamma, const float* beta, 
               float* output, int row_size, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    bool general_calc = tid == 0;

    // Calculating mean

    float local_sum = 0.0f;
    for (int col = tid; col < row_size; col += BLOCK_SIZE) {
        local_sum += input[row * row_size + col];
    }

    __shared__ float local_sums[BLOCK_SIZE];
    __shared__ float sum_in_row;
    local_sums[tid] = local_sum;
    __syncthreads();

    __shared__ float mean;
    if (general_calc) {
        sum_in_row = 0.0f;
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum_in_row += local_sums[i];
        }
        mean = sum_in_row / row_size;
    }

    __syncthreads();

    // Calculating variance

    local_sum = 0.0f;
    float tmp_input;
    for (int col = tid; col < row_size; col += BLOCK_SIZE) {
        tmp_input = input[row * row_size + col];
        local_sum += (tmp_input - mean) * (tmp_input - mean);
    }

    local_sums[tid] = local_sum;
    __syncthreads();


    __shared__ float variance;
    if (general_calc) {
        sum_in_row = 0.0f;
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum_in_row += local_sums[i];
        }
        variance = sum_in_row / row_size;
    }

    __syncthreads();

    // Calculating output
    int index;
    for (int col = tid; col < row_size; col += BLOCK_SIZE) {
        index = row * row_size + col;
        tmp_input = (input[index] - mean) / sqrtf(variance + eps);
        output[index] = gamma[col] * tmp_input + beta[col];
    }
}
''', 'layernormImpl')


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """
    Apply Layer Normalization to each row of the input matrix.

    Parameters
    ----------
    input : list or numpy.ndarray of float
        Flattened matrix in row‑major order. Its length must be divisible by row_size.
    gamma : list or numpy.ndarray of float
        Scale parameter, length = row_size.
    beta : list or numpy.ndarray of float
        Shift parameter, length = row_size.
    row_size : int
        Number of features per row (i.e., number of columns).
    eps : float, optional
        Small constant for numerical stability.

    Returns
    -------
    numpy.ndarray
        Flattened matrix of the same shape as input, containing the row‑wise
        normalized results.
    """

    x = cp.asarray(input)
    g = cp.asarray(gamma)
    b = cp.asarray(beta)
    y = cp.empty_like(x)

    block_size = 32
    row_count = input.size // row_size
    layernormImpl((row_count,), (block_size,), (x, g, b, y, row_size, eps))

    return cp.asnumpy(y)
