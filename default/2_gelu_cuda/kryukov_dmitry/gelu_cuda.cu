#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <cmath>
#include <stdint.h>

#define NO_INIT_VECTOR
#ifdef NO_INIT_VECTOR
template<typename T>
struct vec_ptrs { T* start; T* finish; T* end; };

template<typename T>
static void set_vec_size(std::vector<T>& v, size_t n) {
    reinterpret_cast<vec_ptrs<T>&>(v).finish = v.data() + n;
}
#endif

constexpr float K = 0.044715f;
constexpr float COEFF = 0.7978845608028654f;


//#define FASTAPPROX
#ifdef FASTAPPROX
    // --- fast approximate exp/tanh ---
    // from https://github.com/romeric/fastapprox/blob/master/fastapprox/src/fastexp.h
    __host__ __device__ static inline float
    fastpow2(float p) {
        float offset = (p < 0) ? 1.0f : 0.0f;
        float clipp = (p < -126) ? -126.0f : p;
        int w = clipp;
        float z = clipp - w + offset;
        union { uint32_t i; float f; } v = { static_cast<uint32_t>((1 << 23) * (clipp + 121.2740575f + 27.7280233f / (4.84252568f - z) - 1.49012907f * z)) };

        return v.f;
    }
    __host__ __device__ static inline float
    fastexp(float p) {
        return fastpow2(1.442695040f * p);
    }

    __host__ __device__ static inline float
    exp_tanh(float x) {
        float exp_x = fastexp(-2.0f * fabsf(x));
        return copysignf((1.0f - exp_x) / (1.0f + exp_x), x);
    }
#else
    __host__ __device__ static inline float
    exp_tanh(float x) {
        float exp_x = expf(-2.0f * x);
        return (1.0f - exp_x) / (1.0f + exp_x);
    }
#endif


#ifdef NO_OPT

__global__ void gelu_kernel(const float* input, float* output, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = input[i];
    output[i] = 0.5f * x * (1.0f + tanhf(COEFF * (x + K * x * x * x)));
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t n = input.size();
    size_t bytes = n * sizeof(float);

    float *d_input = nullptr;
    float *d_output = nullptr;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);

    const int block_size = 256;
    const int blocks = (n + block_size - 1) / block_size;
    gelu_kernel<<<blocks, block_size>>>(d_input, d_output, n);
    cudaDeviceSynchronize();

    std::vector<float> output(n);
    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

#else

__global__ void gelu_kernel(const float* input, float* output, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = input[i];
    output[i] = 0.5f * x * (1.0f + exp_tanh(COEFF * (x + K * x * x * x)));
}

class MemManager {
public:
    MemManager() {}
    ~MemManager() {
        if (d_input)  { cudaFree(d_input);  d_input  = nullptr; }
        if (d_output) { cudaFree(d_output); d_output = nullptr; }
        d_bytes = 0;
        if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
    }

    MemManager(const MemManager&) = delete;
    MemManager& operator=(const MemManager&) = delete;

    inline void resize(size_t bytes) {
        if (!stream) cudaStreamCreate(&stream);

        if (bytes == d_bytes) return;

        if (d_input)  { cudaFree(d_input);  d_input  = nullptr; }
        if (d_output) { cudaFree(d_output); d_output = nullptr; }
        d_bytes = 0;

        cudaMalloc(&d_input, bytes);
        cudaMalloc(&d_output, bytes);
        d_bytes = bytes;
    }

    inline float* input()  { return d_input; }
    inline float* output() { return d_output; }
    inline cudaStream_t s() { return stream; }

private:
    float *d_input = nullptr;
    float *d_output = nullptr;
    size_t d_bytes = 0;
    cudaStream_t stream = nullptr;
};

static MemManager mem;

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t n = input.size();
    size_t bytes = n * sizeof(float);
    const int block_size = 256;
    const int blocks = (n + block_size - 1) / block_size;

    mem.resize(bytes);

    cudaMemcpyAsync(mem.input(), input.data(), bytes, cudaMemcpyHostToDevice, mem.s());
    gelu_kernel<<<blocks, block_size, 0, mem.s()>>>(mem.input(), mem.output(), n);

#ifdef NO_INIT_VECTOR
    std::vector<float> output;
    output.reserve(n);
    cudaHostRegister(output.data(), bytes, cudaHostRegisterDefault);
#else
    std::vector<float> output(n);
#endif
    cudaMemcpyAsync(output.data(), mem.output(), bytes, cudaMemcpyDeviceToHost, mem.s());
    cudaStreamSynchronize(mem.s());

#ifdef NO_INIT_VECTOR
    cudaHostUnregister(output.data());
    set_vec_size(output, n);
#endif

    return output;
}

#endif

std::vector<float> GeluSEQ(const std::vector<float>& input) {
    std::vector<float> result(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        result[i] = 0.5f * input[i] * (1.0f + std::tanh(COEFF * (input[i] + K * input[i] * input[i] * input[i])));
    }
    return result;
}
