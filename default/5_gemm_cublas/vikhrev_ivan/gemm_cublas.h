#pragma once

#include <vector>
#include <cmath>

std::vector<float> NaiveGemm(const std::vector<float>& a,
                             const std::vector<float>& b,
                             int n);

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n);

std::vector<float> GemmCUBLASNoPrealloc(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n);
