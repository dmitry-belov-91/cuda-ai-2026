
project(gelu_omp_task1)

cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(gelu_omp_task1 main.cpp gelu_omp.cpp gelu_omp.h)
target_link_libraries(gelu_omp_task1 PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(gelu_omp_task1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(gelu_omp_task1 PRIVATE -O3 -march=native -ffast-math)
