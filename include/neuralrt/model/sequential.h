#pragma once
#include <memory>
#include <vector>
#include "neuralrt/core/tensor.h"

namespace neuralrt::model {

class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& input) const = 0;
};

class LinearLayer : public Layer {
public:
    LinearLayer(size_t in_features, size_t out_features);
    ~LinearLayer() override;
    Tensor forward(const Tensor& input) const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class ReLULayer : public Layer {
public:
    Tensor forward(const Tensor& input) const override;
};

class Sequential {
public:
    Sequential() = default;
    void add(std::unique_ptr<Layer> layer);
    Tensor forward(const Tensor& input) const;
    size_t num_layers() const noexcept;

private:
    std::vector<std::unique_ptr<Layer>> layers_;
};

}  // namespace neuralrt::model