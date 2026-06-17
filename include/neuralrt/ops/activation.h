#pragma once
#include "neuralrt/core/tensor.h"

namespace neuralrt::ops {

void relu_scalar(Tensor& tensor);

#if defined(__AVX2__)
void relu_avx2(Tensor& tensor);
#endif

void relu_inplace(Tensor& tensor);
Tensor relu(const Tensor& input);

}  // namespace neuralrt::ops