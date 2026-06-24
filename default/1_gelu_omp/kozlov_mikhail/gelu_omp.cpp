#include "gelu_omp.h"
#include <omp.h>
#include <cmath>

namespace {
    constexpr float sqrt2pi = 0.7978845608028654f;
    constexpr float coeff = 0.044715f;
    constexpr float half = 0.5f;
    constexpr float one = 1.0f;
    constexpr float two = 2.0f;

    #pragma omp declare simd notinbranch
    inline float fast_tanh(float x) noexcept {
        const float exp_2x = std::exp(two * x);
        return (exp_2x - one) / (exp_2x + one);
    }

    #pragma omp declare simd notinbranch
    inline float gelu_kernel(float x) noexcept {
        const float inner = sqrt2pi * x * (one + coeff * x * x);
        return half * x * (one + fast_tanh(inner));
    }
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t size = input.size();
    std::vector<float> output(size);
    
    const float* __restrict in_ptr = input.data();
    float* __restrict out_ptr = output.data();
    
    #pragma omp parallel for simd schedule(static) aligned(in_ptr, out_ptr : 32)
    for (size_t i = 0; i < size; ++i) {
        out_ptr[i] = gelu_kernel(in_ptr[i]);
    }
    
    return output;
}
