#include "neuralrt/core/shape.h"
#include <numeric>
#include <stdexcept>

namespace neuralrt {

Shape::Shape(std::initializer_list<size_t> dims) : dims_(dims) {}
Shape::Shape(std::vector<size_t> dims) : dims_(std::move(dims)) {}

size_t Shape::rank() const noexcept { return dims_.size(); }

size_t Shape::dim(size_t axis) const {
    if (axis >= dims_.size()) throw std::out_of_range("Shape::dim: axis out of range");
    return dims_[axis];
}

size_t Shape::num_elements() const noexcept {
    if (dims_.empty()) return 0;
    return std::accumulate(dims_.begin(), dims_.end(), static_cast<size_t>(1), std::multiplies<size_t>());
}

const std::vector<size_t>& Shape::dims() const noexcept { return dims_; }
bool Shape::operator==(const Shape& other) const noexcept { return dims_ == other.dims_; }
bool Shape::operator!=(const Shape& other) const noexcept { return !(*this == other); }

std::vector<size_t> Shape::strides() const {
    std::vector<size_t> strides(dims_.size());
    if (dims_.empty()) return strides;
    strides.back() = 1;
    for (size_t i = dims_.size() - 1; i-- > 0;) {
        strides[i] = strides[i + 1] * dims_[i + 1];
    }
    return strides;
}

}  // namespace neuralrt