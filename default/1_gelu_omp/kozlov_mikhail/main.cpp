#include "gelu_omp.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>

constexpr float sqrt2pi = 0.7978845608028654f;
constexpr float coeff = 0.044715f;

std::vector<float> gelu_reference(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + std::tanh(sqrt2pi * x * (1.0f + coeff * x * x)));
    }
    return output;
}

std::vector<float> generate_test_data(size_t size, float min_val = -10.0f, float max_val = 10.0f) {
    std::vector<float> data(size);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
    return data;
}

int main() {
    constexpr size_t test_size = 134217728;
    
    std::cout << "Generating test data..." << std::endl;
    auto input = generate_test_data(test_size);
    
    std::cout << "Running reference implementation..." << std::endl;
    auto reference = gelu_reference(input);
    
    std::cout << "Testing GELU OMP implementation..." << std::endl;
    auto result = GeluOMP(input);
    
    float max_error = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        float error = std::abs(result[i] - reference[i]);
        max_error = std::max(max_error, error);
    }
    
    std::cout << "Max absolute error: " << max_error << std::endl;
    
    if (max_error > 1e-5f) {
        std::cerr << "ERROR: Accuracy test failed!" << std::endl;
        return 1;
    }
    
    std::cout << "Warming up..." << std::endl;
    GeluOMP(input);
    
    std::cout << "Performance measurement..." << std::endl;
    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        GeluOMP(input);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
        std::cout << "  Run " << i+1 << ": " << duration.count() << " s" << std::endl;
    }
    
    double min_time = *std::min_element(time_list.begin(), time_list.end());
    std::cout << "Best time: " << min_time << " s" << std::endl;
    
    std::cout << "Test passed successfully!" << std::endl;
    return 0;
}
