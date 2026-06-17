#include "neuralrt/ops/activation.h"

namespace neuralrt::ops {

void relu_scalar(Tensor& tensor) {
    float* ptr = tensor.data();
    const size_t n = tensor.size();
    for (size_t i = 0; i < n; ++i) ptr[i] = ptr[i] > 0.0f ? ptr[i] : 0.0f;
}

void relu_inplace(Tensor& tensor) {
#if defined(__AVX2__)
    relu_avx2(tensor);
#else
    relu_scalar(tensor);
#endif
}

Tensor relu(const Tensor& input) {
    Tensor result = input.clone();
    relu_inplace(result);
    return result;
}

}  // namespace neuralrt::ops