#include "neuralrt/ops/linear.h"
#include <gtest/gtest.h>
#include <stdexcept>

using neuralrt::Shape;
using neuralrt::Tensor;
using neuralrt::ops::Linear;

TEST(LinearTest, ForwardWithKnownWeightsAndBias) {
    Tensor weight(Shape({2, 2}), {1.0f, 0.0f, 0.0f, 1.0f});
    Tensor bias(Shape({2}), {1.0f, 1.0f});
    Linear linear(weight, bias);
    Tensor input(Shape({1, 2}), {3.0f, 4.0f});
    Tensor output = linear.forward(input);
    EXPECT_FLOAT_EQ(output.at({0, 0}), 4.0f);
    EXPECT_FLOAT_EQ(output.at({0, 1}), 5.0f);
}

TEST(LinearTest, BatchedForward) {
    Tensor weight(Shape({2, 1}), {2.0f, 3.0f});
    Tensor bias(Shape({1}), {0.5f});
    Linear linear(weight, bias);
    Tensor input(Shape({2, 2}), {1.0f, 1.0f, 2.0f, 2.0f});
    Tensor output = linear.forward(input);
    EXPECT_EQ(output.shape().dim(0), 2u);
    EXPECT_EQ(output.shape().dim(1), 1u);
    EXPECT_FLOAT_EQ(output.at({0, 0}), 1.0f * 2.0f + 1.0f * 3.0f + 0.5f);
    EXPECT_FLOAT_EQ(output.at({1, 0}), 2.0f * 2.0f + 2.0f * 3.0f + 0.5f);
}

TEST(LinearTest, ThrowsOnBiasShapeMismatch) {
    Tensor weight(Shape({2, 3})), bad_bias(Shape({2}));
    EXPECT_THROW(Linear(weight, bad_bias), std::invalid_argument);
}

TEST(LinearTest, ThrowsOnIncompatibleInput) {
    Tensor weight(Shape({3, 2})), bias(Shape({2}));
    Linear linear(weight, bias);
    Tensor bad_input(Shape({1, 5}));
    EXPECT_THROW(linear.forward(bad_input), std::invalid_argument);
}

TEST(LinearTest, RandomInitProducesCorrectShapes) {
    Linear linear(4, 8);
    EXPECT_EQ(linear.weight().shape().dim(0), 4u);
    EXPECT_EQ(linear.weight().shape().dim(1), 8u);
    EXPECT_EQ(linear.bias().shape().dim(0), 8u);
}