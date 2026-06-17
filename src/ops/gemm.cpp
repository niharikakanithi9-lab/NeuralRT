#include "neuralrt/ops/gemm.h"

#include <algorithm>
#include <stdexcept>

namespace neuralrt::ops {

namespace detail {
void validate_matmul_shapes(const Tensor& a, const Tensor& b) {
    if (a.shape().rank() != 2 || b.shape().rank() != 2) {
        throw std::invalid_argument("matmul: both operands must be rank-2");
    }
    if (a.shape().dim(1) != b.shape().dim(0)) {
        throw std::invalid_argument("matmul: inner dimensions do not match");
    }
}
}  // namespace detail

Tensor matmul_scalar(const Tensor& a, const Tensor& b) {
    detail::validate_matmul_shapes(a, b);
    const size_t M = a.shape().dim(0);
    const size_t K = a.shape().dim(1);
    const size_t N = b.shape().dim(1);

    Tensor c(Shape({M, N}));
    const float* a_ptr = a.data();
    const float* b_ptr = b.data();
    float* c_ptr = c.data();

    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            const float a_val = a_ptr[i * K + k];
            const float* b_row = b_ptr + k * N;
            float* c_row = c_ptr + i * N;
            for (size_t j = 0; j < N; ++j) c_row[j] += a_val * b_row[j];
        }
    }
    return c;
}

namespace detail {
void matmul_scalar_into(const Tensor& a, const Tensor& b, Tensor& out) {
    const size_t M = a.shape().dim(0);
    const size_t K = a.shape().dim(1);
    const size_t N = b.shape().dim(1);

    float* c_ptr = out.data();
    std::fill(c_ptr, c_ptr + M * N, 0.0f);

    const float* a_ptr = a.data();
    const float* b_ptr = b.data();

    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            const float a_val = a_ptr[i * K + k];
            const float* b_row = b_ptr + k * N;
            float* c_row = c_ptr + i * N;
            for (size_t j = 0; j < N; ++j) c_row[j] += a_val * b_row[j];
        }
    }
}
}  // namespace detail

void matmul_into(const Tensor& a, const Tensor& b, Tensor& out) {
    detail::validate_matmul_shapes(a, b);
    const size_t M = a.shape().dim(0);
    const size_t N = b.shape().dim(1);
    if (out.shape().rank() != 2 || out.shape().dim(0) != M || out.shape().dim(1) != N) {
        throw std::invalid_argument("matmul_into: out shape must be [M, N]");
    }
#if defined(__AVX2__)
    matmul_avx2_into(a, b, out);
#else
    detail::matmul_scalar_into(a, b, out);
#endif
}

Tensor matmul(const Tensor& a, const Tensor& b) {
#if defined(__AVX2__)
    return matmul_avx2(a, b);
#else
    return matmul_scalar(a, b);
#endif
}

bool avx2_available() {
#if defined(__AVX2__)
    return true;
#else
    return false;
#endif
}

}  // namespace neuralrt::ops