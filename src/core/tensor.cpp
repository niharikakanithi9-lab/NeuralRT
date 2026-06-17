#include "neuralrt/core/tensor.h"
#include <random>
#include <stdexcept>

namespace neuralrt {

Tensor::Tensor(Shape shape)
    : shape_(std::move(shape)),
      data_(std::make_shared<std::vector<float>>(shape_.num_elements(), 0.0f)) {}

Tensor::Tensor(Shape shape, std::vector<float> data)
    : shape_(std::move(shape)), data_(std::make_shared<std::vector<float>>(std::move(data))) {
    if (data_->size() != shape_.num_elements()) {
        throw std::invalid_argument("Tensor: data size does not match shape's num_elements");
    }
}

namespace {
size_t flatten_index(const Shape& shape, std::initializer_list<size_t> indices) {
    if (indices.size() != shape.rank()) throw std::invalid_argument("Tensor::at: index rank mismatch");
    const auto strides = shape.strides();
    size_t flat = 0, axis = 0;
    for (size_t idx : indices) {
        if (idx >= shape.dim(axis)) throw std::out_of_range("Tensor::at: index out of bounds");
        flat += idx * strides[axis];
        ++axis;
    }
    return flat;
}
}  // namespace

float& Tensor::at(std::initializer_list<size_t> indices) { return (*data_)[flatten_index(shape_, indices)]; }
float Tensor::at(std::initializer_list<size_t> indices) const { return (*data_)[flatten_index(shape_, indices)]; }
float* Tensor::data() noexcept { return data_->data(); }
const float* Tensor::data() const noexcept { return data_->data(); }
size_t Tensor::size() const noexcept { return data_->size(); }
const Shape& Tensor::shape() const noexcept { return shape_; }

Tensor Tensor::reshape(Shape new_shape) const {
    if (new_shape.num_elements() != shape_.num_elements()) {
        throw std::invalid_argument("Tensor::reshape: element count mismatch, cannot reshape");
    }
    Tensor result;
    result.shape_ = std::move(new_shape);
    result.data_ = data_;
    return result;
}

Tensor Tensor::clone() const { return Tensor(shape_, std::vector<float>(*data_)); }
Tensor Tensor::zeros(Shape shape) { return Tensor(std::move(shape)); }

Tensor Tensor::random(Shape shape, float low, float high) {
    std::vector<float> buf(shape.num_elements());
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(low, high);
    for (auto& v : buf) v = dist(gen);
    return Tensor(std::move(shape), std::move(buf));
}

}  // namespace neuralrt