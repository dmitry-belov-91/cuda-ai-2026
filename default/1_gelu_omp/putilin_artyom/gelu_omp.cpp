#include "gelu_omp.h"

#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input)
{
    const size_t dataLen = input.size();
    std::vector<float> geluVal(dataLen);

    const float const1 = 1.595769121605731; // 2*sqrt(2/pi)
    const float const2 = 0.044715;

    size_t i;
    #pragma omp parallel for simd private(i)
    for (i = 0; i < dataLen; ++i)
    {
        float tanhArg = const1 * (input[i] + const2*input[i]*input[i]*input[i]);
        float expVal  = std::exp(tanhArg);
        float tanhVal = (expVal - 1.0f)/(expVal + 1.0f);
        geluVal[i] = 0.5f * input[i] * (1.0f + tanhVal);
    }

    return geluVal;
}
