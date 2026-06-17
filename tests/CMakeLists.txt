include(FetchContent)
FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(neuralrt_tests
    test_tensor.cpp
    test_gemm.cpp
    test_linear.cpp
    test_activation.cpp
    test_sequential.cpp
    test_simd_correctness.cpp
)

target_link_libraries(neuralrt_tests PRIVATE neuralrt GTest::gtest_main)
target_include_directories(neuralrt_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
include(GoogleTest)
gtest_discover_tests(neuralrt_tests)