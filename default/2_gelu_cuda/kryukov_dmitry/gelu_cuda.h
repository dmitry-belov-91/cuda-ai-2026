#ifndef __GELU_CUDA_H
#define __GELU_CUDA_H

#include <vector>

std::vector<float> GeluCUDA(const std::vector<float>& input);

std::vector<float> GeluSEQ(const std::vector<float>& input);

#endif // __GELU_CUDA_H