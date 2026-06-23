#include "gelu_omp.h"

#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t size = input.size();
    std::vector<float> geluVal(size);

    const float coef0 = -1.595769121605731;
    const float coef1 = 0.044715;

    size_t i;
    #pragma omp parallel for simd private(i)
    for (i = 0; i < size; ++i)
    {
        float x = coef0 * (input[i] + coef1 * input[i] * input[i] * input[i]);
        float expVal  = std::exp(x);
        geluVal[i] = input[i] / (1.0f + expVal);
    }

    return geluVal;
}