cmake_minimum_required(VERSION 3.18)
project(2_gelu_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(gelu_cuda gelu_cuda.cu gelu_cuda.h main.cpp)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(gelu_cuda PRIVATE -O3 -DNDEBUG)
endif()
