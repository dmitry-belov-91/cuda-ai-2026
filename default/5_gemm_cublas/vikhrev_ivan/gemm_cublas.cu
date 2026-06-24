#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <stdexcept>

std::vector<float> NaiveGemm(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Input vectors must have the same size");
    }

    if (a.size() != n * n) {
        throw std::invalid_argument("Input vector size must be equal to n * n");
    }

    std::vector<float> c(a.size(), 0.f);
    for (auto i = 0; i < n; ++i) {
        for (auto j = 0; j < n; ++j) {
            for (auto k = 0; k < n; ++k) {
                c[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }

    return c;
}

inline void check_cuda_error(cudaError_t ret_code, const std::string& message = "") {
    if (ret_code != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(ret_code));
    }
}

inline void check_cublas_error(cublasStatus_t ret_code, const std::string& message = "") {
    if (ret_code != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(message) + ": " + cublasGetStatusString(ret_code));
    }
}

class GemmCublasRunner {
public:
    GemmCublasRunner() {
        check_cublas_error(cublasCreate(&handle));
    }

    ~GemmCublasRunner() {
        cudaFree(a_d_input);
        cudaFree(b_d_input);
        cudaFree(c_d_output);
        check_cublas_error(cublasDestroy(handle));;
    }


    GemmCublasRunner(const GemmCublasRunner&) = delete;
    GemmCublasRunner& operator=(const GemmCublasRunner&) = delete;

    std::vector<float> run(const std::vector<float>& a, const std::vector<float>& b, int n) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Input vectors must have the same size");
        }

        if (a.size() != n * n) {
            throw std::invalid_argument("Input vector size must be equal to n * n");
        }

        int elements_num = a.size();
        ensure_capacity(elements_num);

        std::vector<float> c(elements_num);

        check_cuda_error(cudaMemcpy(
            a_d_input,
            a.data(),
            elements_num * sizeof(float),
            cudaMemcpyHostToDevice
        ), "cudaMemcpy a_d_input failed");

        check_cuda_error(cudaMemcpy(
            b_d_input,
            b.data(),
            elements_num * sizeof(float),
            cudaMemcpyHostToDevice
        ), "cudaMemcpy b_d_input failed");

        float alpha = 1.0f;
        float beta = 0;
        check_cublas_error(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
            &alpha, b_d_input, n, a_d_input, n, &beta, c_d_output, n));

        check_cuda_error(cudaMemcpy(
            c.data(),
            c_d_output,
            num_bytes,
            cudaMemcpyDeviceToHost
        ), "cudaMemcpy device to host failed");

        return c;
    }

private:
    void ensure_capacity(std::size_t requested_size) {
        if (requested_size <= capacity) {
            return;
        }

        check_cuda_error(cudaFree(a_d_input), "cudaFree a_d_input failed");
        check_cuda_error(cudaFree(b_d_input), "cudaFree b_d_input failed");
        check_cuda_error(cudaFree(c_d_output), "cudaFree c_d_output failed");

        num_bytes = requested_size * sizeof(float);

        check_cuda_error(cudaMalloc(&a_d_input, num_bytes), "cudaMalloc d_input failed");
        check_cuda_error(cudaMalloc(&b_d_input, num_bytes), "cudaMalloc d_input failed");
        check_cuda_error(cudaMalloc(&c_d_output, num_bytes), "cudaMalloc d_output failed");

        capacity = requested_size;
    }

    cublasHandle_t handle;
    std::size_t capacity = 0;
    std::size_t num_bytes = 0;

    float* a_d_input = nullptr;
    float* b_d_input = nullptr;
    float* c_d_output = nullptr;
};

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {
    static GemmCublasRunner runner;
    return runner.run(a, b, n);
}

std::vector<float> GemmCUBLASNoPrealloc(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Input vectors must have the same size");
    }

    if (a.size() != n * n) {
        throw std::invalid_argument("Input vector size must be equal to n * n");
    }

    cublasHandle_t handle;
    check_cublas_error(cublasCreate(&handle));

    int elements_num = a.size();
    std::vector<float> c(a.size());

    float* a_d_input = nullptr;
    float* b_d_input = nullptr;
    float* c_d_output = nullptr;

    check_cuda_error(cudaMalloc(&a_d_input, elements_num * sizeof(float)));
    check_cuda_error(cudaMalloc(&b_d_input, elements_num * sizeof(float)));
    check_cuda_error(cudaMalloc(&c_d_output, elements_num * sizeof(float)));

    check_cuda_error(cudaMemcpy(
        a_d_input,
        a.data(),
        elements_num * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    check_cuda_error(cudaMemcpy(
        b_d_input,
        b.data(),
        elements_num * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    float alpha = 1.0f;
    float beta = 0;
    check_cublas_error(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
        &alpha, b_d_input, n, a_d_input, n, &beta, c_d_output, n));

    check_cuda_error(cudaMemcpy(
        c.data(),
        c_d_output,
        elements_num * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    check_cuda_error(cudaFree(a_d_input));
    check_cuda_error(cudaFree(b_d_input));
    check_cuda_error(cudaFree(c_d_output));

    check_cublas_error(cublasDestroy(handle));;

    return c;
}
