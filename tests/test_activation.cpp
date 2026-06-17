#include "neuralrt/ops/activation.h"
#include <gtest/gtest.h>

using neuralrt::Shape;
using neuralrt::Tensor;
using neuralrt::ops::relu;
using neuralrt::ops::relu_scalar;

TEST(ActivationTest, InPlaceZerosNegatives) {
    Tensor t(Shape({4}), {-1.0f, 2.0f, -3.0f, 4.0f});
    relu_scalar(t);
    EXPECT_FLOAT_EQ(t.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(t.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(t.data()[2], 0.0f);
    EXPECT_FLOAT_EQ(t.data()[3], 4.0f);
}

TEST(ActivationTest, OutOfPlaceDoesNotMutateInput) {
    Tensor t(Shape({3}), {-1.0f, 0.0f, 1.0f});
    Tensor result = relu(t);
    EXPECT_FLOAT_EQ(t.data()[0], -1.0f);
    EXPECT_FLOAT_EQ(result.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(result.data()[1], 0.0f);
    EXPECT_FLOAT_EQ(result.data()[2], 1.0f);
}

TEST(ActivationTest, AllPositiveUnchanged) {
    Tensor t(Shape({3}), {1.0f, 2.0f, 3.0f});
    relu_scalar(t);
    EXPECT_FLOAT_EQ(t.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(t.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(t.data()[2], 3.0f);
}

TEST(ActivationTest, ZeroStaysZero) {
    Tensor t(Shape({1}), {0.0f});
    relu_scalar(t);
    EXPECT_FLOAT_EQ(t.data()[0], 0.0f);
}