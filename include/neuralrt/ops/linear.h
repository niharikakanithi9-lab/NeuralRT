#pragma once
#include "neuralrt/core/tensor.h"

namespace neuralrt::ops {

class Linear {
public:
    Linear(size_t in_features, size_t out_features);
    Linear(Tensor weight, Tensor bias);
    Tensor forward(const Tensor& input) const;
    const Tensor& weight() const noexcept;
    const Tensor& bias() const noexcept;

private:
    Tensor weight_;
    Tensor bias_;
};

}  // namespace neuralrt::ops