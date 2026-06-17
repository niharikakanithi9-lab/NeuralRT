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

    // Non-owning view into externally-owned memory. Caller must keep
    // external_data alive and unmoved for the Tensor's lifetime. Used by
    // the graph Executor's memory arena to hand out batch-sized views
    // into preallocated slot buffers without heap allocation.
    static Tensor view(Shape shape, float* external_data);

    // Copies other's contents into this Tensor's existing buffer.
    // Shapes must match exactly. Does not allocate.
    void copy_from(const Tensor& other);

private:
    Shape shape_;
    std::shared_ptr<std::vector<float>> data_;
    float* view_ptr_ = nullptr;
};

}  // namespace neuralrt