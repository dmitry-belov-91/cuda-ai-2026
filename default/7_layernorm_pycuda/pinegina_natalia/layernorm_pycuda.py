import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

#import time

cLayerNormKernel = """

__global__ void rowMeans(float* means, const float* input, int col_size, int row_size)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= col_size) return;

    float sum = 0.0f;

    for (int j = 0; j < row_size; ++j) {
        sum += input[i * row_size + j];
    }
    means[i] = sum / row_size;
}

__global__ void subDif(float* input, int col_size, int row_size, float* mean) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= col_size || j >= row_size) return;

    input[i * row_size + j] -= mean[i];
}

__global__ void getSigma(float* mean, const float* input, int col_size, int row_size)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= col_size) return;

    float varSum = 0.0f;
    float diff = 0.0f;

    for (int k = 0; k < row_size; ++k) {
        diff = input[j * row_size + k];
        varSum += diff * diff;
    }

    mean[j] = varSum / row_size;
}

__global__ void getSqrt(float* input, int row_size, float eps)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= row_size) return;

    input[i] = 1.0f / sqrt(input[i] + eps);
}

__global__ void layerNormKernel(float* input, const float* sigma, const float* gamma, const float* beta,
                                            int col_size, int row_size, float eps)
{
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= col_size || j >= row_size) return;

    float x_upd = sigma[i] * gamma[j];
    input[i * row_size + j] = input[i * row_size + j] * x_upd + beta[j];
}
"""

module = SourceModule(cLayerNormKernel)
pyRowMeans = module.get_function("rowMeans")
pySubDif = module.get_function("subDif")
pyGetSigma = module.get_function("getSigma")
pyGetSqrt = module.get_function("getSqrt")
pyLayerNorm = module.get_function("layerNormKernel")

def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):

    x = np.asarray(input, dtype=np.float32)
    gamma = np.asarray(gamma, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)

    col_size = np.int32(x.size / row_size)
    row_size = np.int32(row_size)

    x_gpu = drv.mem_alloc(x.nbytes)
    mean_gpu = drv.mem_alloc(int(col_size * 4))
    gamma_gpu = drv.mem_alloc(gamma.nbytes)
    beta_gpu = drv.mem_alloc(beta.nbytes)

    drv.memcpy_htod(x_gpu, x)
    drv.memcpy_htod(gamma_gpu, gamma)
    drv.memcpy_htod(beta_gpu, beta)

    bs_vec = (256, 1, 1)
    nb_vec = (int((col_size + 255) // 256), 1)

    bs_mtrx = (16, 16, 1)
    nb_mtrx = (int((row_size + 15) // 16), int((col_size + 15) // 16),1)

    pyRowMeans(mean_gpu, x_gpu, col_size, row_size, block=bs_vec, grid=nb_vec)
    pySubDif(x_gpu, col_size, row_size, mean_gpu, block=bs_mtrx, grid=nb_mtrx)
    pyGetSigma(mean_gpu, x_gpu, col_size, row_size, block=bs_vec, grid=nb_vec)
    pyGetSqrt(mean_gpu, col_size, np.float32(eps), block=bs_vec, grid=nb_vec)
    pyLayerNorm(x_gpu, mean_gpu, gamma_gpu, beta_gpu, col_size, row_size, np.float32(eps),block=bs_mtrx, grid=nb_mtrx)

    y = np.empty_like(x)
    drv.memcpy_dtoh(y, x_gpu)

    return y

# --- Test ---
#start_time = time.perf_counter()

#batch_size = 8192
#row_size =8192

#x_test = np.full((batch_size, row_size),300)
#x_test = [[i + j * row_size for i in range(row_size)] for j in range(row_size)]
#gamma_test = np.ones(row_size, dtype=np.float32)
#beta_test = np.zeros(row_size, dtype=np.float32)

#y_out = layernorm_pycuda(x_test, gamma_test, beta_test, row_size)

#print(y_out)

#end_time = time.perf_counter()
#execution_time = end_time - start_time
#print(f"Время выполнения: {execution_time:.6f} секунд")

#print(y_out)
#print(y_out)
