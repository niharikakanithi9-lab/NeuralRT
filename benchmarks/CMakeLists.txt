include(FetchContent)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(googlebenchmark URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.tar.gz)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(neuralrt_benchmarks
    bench_gemm.cpp
    bench_sequential.cpp
    bench_activation.cpp
)
target_link_libraries(neuralrt_benchmarks PRIVATE neuralrt benchmark::benchmark)
target_include_directories(neuralrt_benchmarks PRIVATE ${CMAKE_SOURCE_DIR}/include)