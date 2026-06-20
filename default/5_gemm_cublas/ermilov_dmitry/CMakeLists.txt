cmake_minimum_required(VERSION 3.10)

project(5_gemm_cublas LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


find_package(CUDAToolkit REQUIRED)

add_executable(gemm_cublas main.cpp gemm_cublas.cu)
target_link_libraries(gemm_cublas PRIVATE CUDA::cublas)