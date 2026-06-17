#pragma once
#include <cstddef>
#include <initializer_list>
#include <vector>

namespace neuralrt {

class Shape {
public:
    Shape() = default;
    Shape(std::initializer_list<size_t> dims);
    explicit Shape(std::vector<size_t> dims);

    size_t rank() const noexcept;
    size_t dim(size_t axis) const;
    size_t num_elements() const noexcept;
    const std::vector<size_t>& dims() const noexcept;

    bool operator==(const Shape& other) const noexcept;
    bool operator!=(const Shape& other) const noexcept;
    std::vector<size_t> strides() const;

private:
    std::vector<size_t> dims_;
};

}  // namespace neuralrt