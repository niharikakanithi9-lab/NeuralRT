#pragma once
#include "neuralrt/core/tensor.h"

namespace neuralrt::ops {

Tensor matmul_scalar(const Tensor& a, const Tensor& b);

#if defined(__AVX2__)
Tensor matmul_avx2(const Tensor& a, const Tensor& b);
#endif

Tensor matmul(const Tensor& a, const Tensor& b);
bool avx2_available();

namespace detail {
void validate_matmul_shapes(const Tensor& a, const Tensor& b);
}

}  // namespace neuralrt::ops