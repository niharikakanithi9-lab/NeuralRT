#include "neuralrt/ops/gemm.h"
#include <gtest/gtest.h>
#include <stdexcept>

using neuralrt::Shape;
using neuralrt::Tensor;
using neuralrt::ops::matmul_scalar;

TEST(GemmTest, IdentityMultiplication) {
    Tensor a(Shape({2, 2}), {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor identity(Shape({2, 2}), {1.0f, 0.0f, 0.0f, 1.0f});
    Tensor c = matmul_scalar(a, identity);
    EXPECT_FLOAT_EQ(c.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(c.at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(c.at({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 4.0f);
}

TEST(GemmTest, KnownResultRectangular) {
    Tensor a(Shape({2, 3}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor b(Shape({3, 2}), {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    Tensor c = matmul_scalar(a, b);
    EXPECT_FLOAT_EQ(c.at({0, 0}), 58.0f);
    EXPECT_FLOAT_EQ(c.at({0, 1}), 64.0f);
    EXPECT_FLOAT_EQ(c.at({1, 0}), 139.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 154.0f);
}

TEST(GemmTest, ThrowsOnDimensionMismatch) {
    Tensor a(Shape({2, 3})), b(Shape({4, 2}));
    EXPECT_THROW(matmul_scalar(a, b), std::invalid_argument);
}

TEST(GemmTest, ThrowsOnNonRank2Input) {
    Tensor a(Shape({2, 3, 1})), b(Shape({3, 2}));
    EXPECT_THROW(matmul_scalar(a, b), std::invalid_argument);
}

TEST(GemmTest, ResultShapeIsCorrect) {
    Tensor a(Shape({5, 7})), b(Shape({7, 3}));
    Tensor c = matmul_scalar(a, b);
    EXPECT_EQ(c.shape().dim(0), 5u);
    EXPECT_EQ(c.shape().dim(1), 3u);
}

TEST(GemmTest, ZeroMatrixProducesZero) {
    Tensor a = Tensor::zeros(Shape({3, 3}));
    Tensor b = Tensor::random(Shape({3, 3}));
    Tensor c = matmul_scalar(a, b);
    for (size_t i = 0; i < c.size(); ++i) EXPECT_FLOAT_EQ(c.data()[i], 0.0f);
}