#include "neuralrt/ops/activation.h"

#if defined(__AVX2__)
#include <immintrin.h>

namespace neuralrt::ops {

void relu_avx2(Tensor& tensor) {
    float* ptr = tensor.data();
    const size_t n = tensor.size();
    const __m256 zero = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(ptr + i);
        v = _mm256_max_ps(v, zero);
        _mm256_storeu_ps(ptr + i, v);
    }
    for (; i < n; ++i) ptr[i] = ptr[i] > 0.0f ? ptr[i] : 0.0f;
}

}  // namespace neuralrt::ops
#endif