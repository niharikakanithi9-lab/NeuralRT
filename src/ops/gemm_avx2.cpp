#include "neuralrt/ops/gemm.h"

#if defined(__AVX2__)
#include <algorithm>
#include <immintrin.h>

namespace neuralrt::ops {

namespace {
constexpr size_t kBlockM = 64;
constexpr size_t kBlockK = 256;
constexpr size_t kBlockN = 256;

inline void accumulate_strip(const float* a_row, const float* b_ptr, float* c_strip, size_t N,
                              size_t k_start, size_t k_end, size_t j) {
    __m256 acc = _mm256_loadu_ps(c_strip);
    for (size_t k = k_start; k < k_end; ++k) {
        const __m256 a_vec = _mm256_set1_ps(a_row[k]);
        const __m256 b_vec = _mm256_loadu_ps(b_ptr + k * N + j);
        acc = _mm256_fmadd_ps(a_vec, b_vec, acc);
    }
    _mm256_storeu_ps(c_strip, acc);
}

inline void accumulate_scalar_tail(const float* a_row, const float* b_ptr, float* c_ptr, size_t N,
                                    size_t k_start, size_t k_end, size_t j) {
    float acc = c_ptr[j];
    for (size_t k = k_start; k < k_end; ++k) acc += a_row[k] * b_ptr[k * N + j];
    c_ptr[j] = acc;
}
}  // namespace

Tensor matmul_avx2(const Tensor& a, const Tensor& b) {
    detail::validate_matmul_shapes(a, b);
    const size_t M = a.shape().dim(0);
    const size_t K = a.shape().dim(1);
    const size_t N = b.shape().dim(1);

    Tensor c(Shape({M, N}));
    const float* a_ptr = a.data();
    const float* b_ptr = b.data();
    float* c_ptr = c.data();

    for (size_t i0 = 0; i0 < M; i0 += kBlockM) {
        const size_t i_end = std::min(i0 + kBlockM, M);
        for (size_t n0 = 0; n0 < N; n0 += kBlockN) {
            const size_t n_end = std::min(n0 + kBlockN, N);
            for (size_t k0 = 0; k0 < K; k0 += kBlockK) {
                const size_t k_end = std::min(k0 + kBlockK, K);
                for (size_t i = i0; i < i_end; ++i) {
                    const float* a_row = a_ptr + i * K;
                    float* c_row = c_ptr + i * N;
                    size_t j = n0;
                    for (; j + 8 <= n_end; j += 8) accumulate_strip(a_row, b_ptr, c_row + j, N, k0, k_end, j);
                    for (; j < n_end; ++j) accumulate_scalar_tail(a_row, b_ptr, c_ptr + i * N, N, k0, k_end, j);
                }
            }
        }
    }
    return c;
}

void matmul_avx2_into(const Tensor& a, const Tensor& b, Tensor& out) {
    detail::validate_matmul_shapes(a, b);
    const size_t M = a.shape().dim(0);
    const size_t K = a.shape().dim(1);
    const size_t N = b.shape().dim(1);

    const float* a_ptr = a.data();
    const float* b_ptr = b.data();
    float* c_ptr = out.data();
    std::fill(c_ptr, c_ptr + M * N, 0.0f);

    for (size_t i0 = 0; i0 < M; i0 += kBlockM) {
        const size_t i_end = std::min(i0 + kBlockM, M);
        for (size_t n0 = 0; n0 < N; n0 += kBlockN) {
            const size_t n_end = std::min(n0 + kBlockN, N);
            for (size_t k0 = 0; k0 < K; k0 += kBlockK) {
                const size_t k_end = std::min(k0 + kBlockK, K);
                for (size_t i = i0; i < i_end; ++i) {
                    const float* a_row = a_ptr + i * K;
                    float* c_row = c_ptr + i * N;
                    size_t j = n0;
                    for (; j + 8 <= n_end; j += 8) accumulate_strip(a_row, b_ptr, c_row + j, N, k0, k_end, j);
                    for (; j < n_end; ++j) accumulate_scalar_tail(a_row, b_ptr, c_ptr + i * N, N, k0, k_end, j);
                }
            }
        }
    }
}

}  // namespace neuralrt::ops
#endif