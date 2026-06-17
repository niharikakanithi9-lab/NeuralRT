#pragma once
#include "neuralrt/core/tensor.h"

namespace neuralrt::ops {

Tensor matmul_scalar(const Tensor& a, const Tensor& b);

#if defined(__AVX2__)
Tensor matmul_avx2(const Tensor& a, const Tensor& b);
void matmul_avx2_into(const Tensor& a, const Tensor& b, Tensor& out);
#endif

Tensor matmul(const Tensor& a, const Tensor& b);

// Zero-allocation variant: writes C = A * B into caller-provided `out`
// (must already be shaped [M, N]) instead of allocating a new Tensor.
void matmul_into(const Tensor& a, const Tensor& b, Tensor& out);

bool avx2_available();

namespace detail {
void validate_matmul_shapes(const Tensor& a, const Tensor& b);
}

}  // namespace neuralrt::ops