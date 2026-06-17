#pragma once
#include <initializer_list>
#include <memory>
#include <vector>
#include "neuralrt/core/shape.h"

namespace neuralrt {

class Tensor {
public:
    Tensor() = default;
    explicit Tensor(Shape shape);
    Tensor(Shape shape, std::vector<float> data);

    float& at(std::initializer_list<size_t> indices);
    float at(std::initializer_list<size_t> indices) const;

    float* data() noexcept;
    const float* data() const noexcept;
    size_t size() const noexcept;
    const Shape& shape() const noexcept;

    Tensor reshape(Shape new_shape) const;
    Tensor clone() const;

    static Tensor zeros(Shape shape);
    static Tensor random(Shape shape, float low = -1.0f, float high = 1.0f);

private:
    Shape shape_;
    std::shared_ptr<std::vector<float>> data_;
};

}  // namespace neuralrt