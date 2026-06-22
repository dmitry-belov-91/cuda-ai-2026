import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

from pycuda.compiler import SourceModule

_BLOCK_SIZE = 256

_KERNEL_CODE = r"""
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

__global__ void LayerNormCUDAKernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int row_size,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;

    const float* input_row = input + row * row_size;
    float* output_row = output + row * row_size;
    
    __shared__ float shared_sum[WARPS_PER_BLOCK];
    __shared__ float shared_sumsq[WARPS_PER_BLOCK];

    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    
    for (int i = tid * 4; i < row_size; i += blockDim.x * 4) {
        float4 group_4_floats = reinterpret_cast<const float4*>(input_row + i)[0];
        
        local_sum += group_4_floats.x + group_4_floats.y + group_4_floats.z + group_4_floats.w;
        local_sumsq += group_4_floats.x * group_4_floats.x + group_4_floats.y * group_4_floats.y + group_4_floats.z * group_4_floats.z + group_4_floats.w * group_4_floats.w;
    }
    
    for (int range = WARP_SIZE / 2; range > 0; range >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, range);
        local_sumsq += __shfl_down_sync(0xffffffff, local_sumsq, range);
    }
    
    if (lane == 0) {
        shared_sum[warp_id] = local_sum;
        shared_sumsq[warp_id] = local_sumsq;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float warp_sum = (lane < blockDim.x / WARP_SIZE) ? shared_sum[lane] : 0.0f;
        float warp_sumsq = (lane < blockDim.x / WARP_SIZE) ? shared_sumsq[lane] : 0.0f;
        
        for (int range = WARP_SIZE / 2; range > 0; range >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, range);
            warp_sumsq += __shfl_down_sync(0xffffffff, warp_sumsq, range);
        }
        
        if (lane == 0) {
            shared_sum[0] = warp_sum;
            shared_sumsq[0] = warp_sumsq;
        }
    }
    __syncthreads();
    
    float mean = shared_sum[0] / row_size;
    float variance = shared_sumsq[0] / row_size - mean * mean;
    float r_variance = rsqrtf(variance + eps);
    
    for (int i = tid * 4; i < row_size; i += blockDim.x * 4) {
        float4 group_4_floats = reinterpret_cast<const float4*>(input_row + i)[0];
        float4 gamma_group_4_floats = reinterpret_cast<const float4*>(gamma + i)[0];
        float4 beta_group_4_floats = reinterpret_cast<const float4*>(beta + i)[0];
        
        float4 out_group_4_floats;
        out_group_4_floats.x = ((group_4_floats.x - mean) * r_variance) * gamma_group_4_floats.x + beta_group_4_floats.x;
        out_group_4_floats.y = ((group_4_floats.y - mean) * r_variance) * gamma_group_4_floats.y + beta_group_4_floats.y;
        out_group_4_floats.z = ((group_4_floats.z - mean) * r_variance) * gamma_group_4_floats.z + beta_group_4_floats.z;
        out_group_4_floats.w = ((group_4_floats.w - mean) * r_variance) * gamma_group_4_floats.w + beta_group_4_floats.w;
        
        reinterpret_cast<float4*>(output_row + i)[0] = out_group_4_floats;
    }
}
"""

_mod = SourceModule(_KERNEL_CODE, options=["-O3", "-use_fast_math"])
_layernorm_cuda_kernel = _mod.get_function("LayerNormCUDAKernel")

def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    input_on_cpu = np.asarray(input, dtype=np.float32).ravel()
    gamma_on_cpu = np.asarray(gamma, dtype=np.float32).ravel()
    beta_on_cpu = np.asarray(beta, dtype=np.float32).ravel()
    output_on_cpu = np.empty_like(input_on_cpu)

    nelem = input_on_cpu.size
    nelem_bytes = nelem * 4
    row_size_bytes = row_size * 4
    
    input_on_gpu = cuda.mem_alloc(nelem_bytes)
    output_on_gpu = cuda.mem_alloc(nelem_bytes)
    gamma_on_gpu = cuda.mem_alloc(row_size_bytes)
    beta_on_gpu = cuda.mem_alloc(row_size_bytes)
    
    cuda.memcpy_htod(input_on_gpu, input_on_cpu)
    cuda.memcpy_htod(gamma_on_gpu, gamma_on_cpu)
    cuda.memcpy_htod(beta_on_gpu, beta_on_cpu)

    _layernorm_cuda_kernel(
        input_on_gpu, gamma_on_gpu, beta_on_gpu, output_on_gpu,
        np.int32(row_size), np.float32(eps),
        block=(_BLOCK_SIZE, 1, 1),
        grid=(input_on_cpu.size // row_size, 1),
    )

    cuda.memcpy_dtoh(output_on_cpu, output_on_gpu)
    return output_on_cpu
