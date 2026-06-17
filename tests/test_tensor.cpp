#include "neuralrt/core/tensor.h"
#include <gtest/gtest.h>
#include <stdexcept>

using neuralrt::Shape;
using neuralrt::Tensor;

TEST(TensorTest, ConstructsWithZeros) {
    Tensor t(Shape({2, 3}));
    EXPECT_EQ(t.size(), 6u);
    for (size_t i = 0; i < t.size(); ++i) EXPECT_FLOAT_EQ(t.data()[i], 0.0f);
}

TEST(TensorTest, ConstructsFromData) {
    Tensor t(Shape({2, 2}), {1.0f, 2.0f, 3.0f, 4.0f});
    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(t.at({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(t.at({1, 1}), 4.0f);
}

TEST(TensorTest, ConstructorThrowsOnSizeMismatch) {
    EXPECT_THROW(Tensor(Shape({2, 2}), {1.0f, 2.0f}), std::invalid_argument);
}

TEST(TensorTest, AtThrowsOnOutOfBounds) {
    Tensor t(Shape({2, 2}));
    EXPECT_THROW(t.at({2, 0}), std::out_of_range);
}

TEST(TensorTest, MutationThroughAt) {
    Tensor t(Shape({2, 2}));
    t.at({0, 0}) = 42.0f;
    EXPECT_FLOAT_EQ(t.at({0, 0}), 42.0f);
}

TEST(TensorTest, ReshapeIsZeroCopy) {
    Tensor t(Shape({2, 3}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor r = t.reshape(Shape({3, 2}));
    EXPECT_EQ(r.shape().dim(0), 3u);
    EXPECT_EQ(r.shape().dim(1), 2u);
    t.at({0, 0}) = 99.0f;
    EXPECT_FLOAT_EQ(r.data()[0], 99.0f);
}

TEST(TensorTest, ReshapeThrowsOnElementCountMismatch) {
    Tensor t(Shape({2, 3}));
    EXPECT_THROW(t.reshape(Shape({4, 4})), std::invalid_argument);
}

TEST(TensorTest, CloneIsDeepCopy) {
    Tensor t(Shape({2, 2}), {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor c = t.clone();
    t.at({0, 0}) = 100.0f;
    EXPECT_FLOAT_EQ(c.at({0, 0}), 1.0f);
}

TEST(TensorTest, ZerosFactory) {
    Tensor t = Tensor::zeros(Shape({3}));
    for (size_t i = 0; i < t.size(); ++i) EXPECT_FLOAT_EQ(t.data()[i], 0.0f);
}

TEST(TensorTest, RandomFactoryRespectsBounds) {
    Tensor t = Tensor::random(Shape({100}), -1.0f, 1.0f);
    for (size_t i = 0; i < t.size(); ++i) {
        EXPECT_GE(t.data()[i], -1.0f);
        EXPECT_LE(t.data()[i], 1.0f);
    }
}

TEST(ShapeTest, NumElementsComputesProduct) {
    Shape s({2, 3, 4});
    EXPECT_EQ(s.num_elements(), 24u);
}

TEST(ShapeTest, StridesAreRowMajor) {
    Shape s({2, 3, 4});
    auto strides = s.strides();
    EXPECT_EQ(strides[0], 12u);
    EXPECT_EQ(strides[1], 4u);
    EXPECT_EQ(strides[2], 1u);
}

TEST(ShapeTest, EqualityOperator) {
    Shape a({2, 3}), b({2, 3}), c({3, 2});
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}