#include "neuralrt/ops/linear.h"

#include <stdexcept>

#include "neuralrt/ops/gemm.h"

namespace neuralrt::ops {

Linear::Linear(size_t in_features, size_t out_features)
    : weight_(Tensor::random(Shape({in_features, out_features}), -0.1f, 0.1f)),
      bias_(Tensor::zeros(Shape({out_features}))) {}

Linear::Linear(Tensor weight, Tensor bias) : weight_(std::move(weight)), bias_(std::move(bias)) {
    if (weight_.shape().rank() != 2) {
        throw std::invalid_argument("Linear: weight must be rank-2 [in_features, out_features]");
    }
    if (bias_.shape().rank() != 1 || bias_.shape().dim(0) != weight_.shape().dim(1)) {
        throw std::invalid_argument("Linear: bias shape must match weight's out_features");
    }
}

Tensor Linear::forward(const Tensor& input) const {
    if (input.shape().rank() != 2 || input.shape().dim(1) != weight_.shape().dim(0)) {
        throw std::invalid_argument("Linear::forward: input shape incompatible with weight");
    }
    Tensor out = matmul(input, weight_);
    const size_t batch = out.shape().dim(0);
    const size_t out_features = out.shape().dim(1);
    float* out_ptr = out.data();
    const float* bias_ptr = bias_.data();
    for (size_t i = 0; i < batch; ++i) {
        float* row = out_ptr + i * out_features;
        for (size_t j = 0; j < out_features; ++j) row[j] += bias_ptr[j];
    }
    return out;
}

void Linear::forward_into(const Tensor& input, Tensor& output) const {
    if (input.shape().rank() != 2 || input.shape().dim(1) != weight_.shape().dim(0)) {
        throw std::invalid_argument("Linear::forward_into: input shape incompatible with weight");
    }
    const size_t batch = input.shape().dim(0);
    const size_t out_features = weight_.shape().dim(1);
    if (output.shape().rank() != 2 || output.shape().dim(0) != batch ||
        output.shape().dim(1) != out_features) {
        throw std::invalid_argument("Linear::forward_into: output shape must be [batch, out_features]");
    }
    matmul_into(input, weight_, output);
    float* out_ptr = output.data();
    const float* bias_ptr = bias_.data();
    for (size_t i = 0; i < batch; ++i) {
        float* row = out_ptr + i * out_features;
        for (size_t j = 0; j < out_features; ++j) row[j] += bias_ptr[j];
    }
}

const Tensor& Linear::weight() const noexcept { return weight_; }
const Tensor& Linear::bias() const noexcept { return bias_; }

}  // namespace neuralrt::ops