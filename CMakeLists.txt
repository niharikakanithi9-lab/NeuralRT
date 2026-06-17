cmake_minimum_required(VERSION 3.16)
project(NeuralRT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(NEURALRT_BUILD_TESTS "Build unit tests" ON)
option(NEURALRT_BUILD_BENCHMARKS "Build benchmarks" ON)
option(NEURALRT_ENABLE_AVX2 "Enable AVX2 SIMD codepaths" ON)

include(cmake/CompilerWarnings.cmake)

add_library(neuralrt
    src/core/shape.cpp
    src/core/tensor.cpp
    src/ops/gemm.cpp
    src/ops/linear.cpp
    src/ops/activation.cpp
    src/model/sequential.cpp
)

target_include_directories(neuralrt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(neuralrt PUBLIC cxx_std_17)
neuralrt_set_warnings(neuralrt)

set_source_files_properties(
    src/ops/gemm.cpp
    src/ops/activation.cpp
    PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize;-fno-tree-slp-vectorize"
)

if(NEURALRT_ENABLE_AVX2)
    target_sources(neuralrt PRIVATE
        src/ops/gemm_avx2.cpp
        src/ops/activation_avx2.cpp
    )
    target_compile_options(neuralrt PUBLIC -mavx2 -mfma)
endif()

if(NEURALRT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(NEURALRT_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()