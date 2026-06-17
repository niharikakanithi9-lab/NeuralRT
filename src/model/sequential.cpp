#include "neuralrt/model/sequential.h"
#include "neuralrt/ops/activation.h"
#include "neuralrt/ops/linear.h"

namespace neuralrt::model {

class LinearLayer::Impl {
public:
    Impl(size_t in_features, size_t out_features) : linear_(in_features, out_features) {}
    ops::Linear linear_;
};

LinearLayer::LinearLayer(size_t in_features, size_t out_features)
    : impl_(std::make_unique<Impl>(in_features, out_features)) {}
LinearLayer::~LinearLayer() = default;
Tensor LinearLayer::forward(const Tensor& input) const { return impl_->linear_.forward(input); }
Tensor ReLULayer::forward(const Tensor& input) const { return ops::relu(input); }

void Sequential::add(std::unique_ptr<Layer> layer) { layers_.push_back(std::move(layer)); }

Tensor Sequential::forward(const Tensor& input) const {
    Tensor current = input;
    for (const auto& layer : layers_) current = layer->forward(current);
    return current;
}

size_t Sequential::num_layers() const noexcept { return layers_.size(); }

}  // namespace neuralrt::model