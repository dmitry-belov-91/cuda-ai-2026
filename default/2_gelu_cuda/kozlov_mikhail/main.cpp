#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <cuda_runtime.h>
#include "gelu_cuda.h"

namespace {
    constexpr float SQRT_2_O_PI = 0.7978845608028654f;
    constexpr float COEFF = 0.044715f;
    constexpr size_t N = 134217728;

    std::vector<float> GeluCPU(const std::vector<float>& data) {
        std::vector<float> output(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            float val = data[i];
            float inner = SQRT_2_O_PI * val * (1.0f + COEFF * val * val);
            output[i] = 0.5f * val * (1.0f + std::tanh(inner));
        }
        return output;
    }

    float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
        float diff = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            float current = std::abs(a[i] - b[i]);
            if (current > diff) diff = current;
        }
        return diff;
    }
}

int main() {
    std::cout << "Tastk #2: GELU CUDA" << std::endl;
    std::cout << "Input size: " << N << " elements (" << N * sizeof(float) / (1024.0 * 1024.0) << " MB)" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    std::vector<float> input(N);
    {
        std::mt19937 gen(12345);
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        for (auto& x : input) x = dist(gen);
    }

    std::cout << "\n- CPU reference ..." << std::flush;
    auto cpu_result = GeluCPU(input);
    std::cout << " Done" << std::endl;

    std::cout << "- Computing CUDA result..." << std::flush;
    auto gpu_result = GeluCUDA(input);
    std::cout << " Done" << std::endl;

    float error = MaxAbsDiff(cpu_result, gpu_result);
    std::cout << "[3] Max absolute error: " << error << std::endl;

    if (error < 1e-4f) {
        std::cout << "Status: CORRECT" << std::endl;
    } else {
        std::cout << "Status: INCORRECT" << std::endl;
        return 1;
    }

    std::cout << "\n[4] Performance benchmark (4 iterations)..." << std::endl;
    std::vector<double> timings;
    
    GeluCUDA(input);
    cudaDeviceSynchronize();

    for (int iter = 0; iter < 4; ++iter) {
        auto t_start = std::chrono::high_resolution_clock::now();
        GeluCUDA(input);
        cudaDeviceSynchronize();
        auto t_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(t_end - t_start).count();
        timings.push_back(duration);
        std::cout << "    Iter " << (iter + 1) << ": " << duration << "s" << std::endl;
    }

    auto best = *std::min_element(timings.begin(), timings.end());
    auto avg = std::accumulate(timings.begin(), timings.end(), 0.0) / 4.0;

    std::cout << "\n- Performance results:" << std::endl;
    std::cout << "Best: " << best << "s, avg: " << avg << "s" << std::endl;

    return 0;
}
