import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

from pycuda.compiler import SourceModule

kernels = SourceModule("""
__global__ void get_means(float* input, int samples, int features, float* row_means) {
    int sample_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_index >= samples) return;

    float sum = 0.f;
    for (int j = 0; j < features; ++j) {
        sum += input[sample_index * features + j];
    }
    row_means[sample_index] = sum / features;
}

__global__ void sub_mean(float* input, int samples, int features, float* means) {
    int sample_index = blockIdx.y * blockDim.y + threadIdx.y;
    int feature_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_index >= samples || feature_index >= features) return;

    input[sample_index * features + feature_index] -= means[sample_index];
}

__global__ void get_vars(float* input, int samples, int features, float* row_vars) {
    int sample_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_index >= samples) return;

    float sum = 0.f;
    float x;
    for (int j = 0; j < features; ++j) {
        x = input[sample_index * features + j];
        sum += x * x;
    }
    row_vars[sample_index] = sum / features;
}

__global__ void inv_sqrt_vars(float* vars, int samples, float eps) {
    int sample_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_index >= samples) return;

    vars[sample_index] = 1.f / sqrt(vars[sample_index] + eps);
}

__global__ void layer_norm(float* input, int samples, int features, float* inv_sqrt_vars, float* gamma, float* beta) {
    int sample_index = blockIdx.y * blockDim.y + threadIdx.y;
    int feature_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_index >= samples || feature_index >= features) return;

    float x = inv_sqrt_vars[sample_index] * gamma[feature_index];
    input[sample_index * features + feature_index] = input[sample_index * features + feature_index] * x + beta[feature_index];
}
""")


get_means = kernels.get_function("get_means")
sub_mean = kernels.get_function("sub_mean")
get_vars = kernels.get_function("get_vars")
inv_sqrt_vars = kernels.get_function("inv_sqrt_vars")
layer_norm = kernels.get_function("layer_norm")

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
    input_cpu = np.asarray(input, dtype=np.float32)
    gamma_cpu = np.asarray(gamma, dtype=np.float32)
    beta_cpu = np.asarray(beta, dtype=np.float32)

    samples = np.int32(input_cpu.size / row_size)
    features = np.int32(row_size)

    input_gpu = cuda.mem_alloc(input_cpu.nbytes)
    cuda.memcpy_htod(input_gpu, input_cpu)

    gamma_gpu = cuda.mem_alloc(gamma_cpu.nbytes)
    cuda.memcpy_htod(gamma_gpu, gamma_cpu)

    beta_gpu = cuda.mem_alloc(beta_cpu.nbytes)
    cuda.memcpy_htod(beta_gpu, beta_cpu)

    means_gpu = cuda.mem_alloc(int(samples * 4))

    block_size_1d = (256, 1, 1)
    num_blocks_1d = (int((samples + block_size_1d[0] - 1) // block_size_1d[0]), 1)

    block_size_2d = (16, 16, 1)
    num_blocks_2d = (
        int((features + block_size_2d[0] - 1) // block_size_2d[0]),
        int((samples + block_size_2d[1] - 1) // block_size_2d[1]),
        1,
    )

    get_means(
        input_gpu,
        samples,
        features,
        means_gpu,
        block=block_size_1d,
        grid=num_blocks_1d,
    )

    sub_mean(
        input_gpu,
        samples,
        features,
        means_gpu,
        block=block_size_2d,
        grid=num_blocks_2d,
    )

    get_vars(
        input_gpu,
        samples,
        features,
        means_gpu,
        block=block_size_1d,
        grid=num_blocks_1d,
    )

    inv_sqrt_vars(
        means_gpu,
        samples,
        np.float32(eps),
        block=block_size_1d,
        grid=num_blocks_1d,
    )

    layer_norm(
        input_gpu,
        samples,
        features,
        means_gpu,
        gamma_gpu,
        beta_gpu,
        block=block_size_2d,
        grid=num_blocks_2d,
    )

    out = np.empty_like(input_cpu)
    cuda.memcpy_dtoh(out, input_gpu)

    input_gpu.free()
    gamma_gpu.free()
    beta_gpu.free()
    means_gpu.free()

    return out
