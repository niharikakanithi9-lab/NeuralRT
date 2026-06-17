#include "neuralrt/ops/gemm.h"
#include <gtest/gtest.h>
#include "neuralrt/ops/activation.h"

using neuralrt::Shape;
using neuralrt::Tensor;
using neuralrt::ops::matmul;
using neuralrt::ops::matmul_scalar;
using neuralrt::ops::relu;
using neuralrt::ops::relu_scalar;

namespace {
constexpr float kGemmTolerance = 1e-4f;

void ExpectTensorsClose(const Tensor& a, const Tensor& b, float tol) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) EXPECT_NEAR(a.data()[i], b.data()[i], tol) << "mismatch at " << i;
}
}  // namespace

#if defined(__AVX2__)
using neuralrt::ops::matmul_avx2;
using neuralrt::ops::relu_avx2;

TEST(GemmAvx2Test, MatchesScalarOnSquareMatrix) {
    Tensor a = Tensor::random(Shape({64, 64}), -1.0f, 1.0f);
    Tensor b = Tensor::random(Shape({64, 64}), -1.0f, 1.0f);
    ExpectTensorsClose(matmul_scalar(a, b), matmul_avx2(a, b), kGemmTolerance);
}

TEST(GemmAvx2Test, MatchesScalarOnNonMultipleOf8Dims) {
    Tensor a = Tensor::random(Shape({37, 53}), -1.0f, 1.0f);
    Tensor b = Tensor::random(Shape({53, 19}), -1.0f, 1.0f);
    ExpectTensorsClose(matmul_scalar(a, b), matmul_avx2(a, b), kGemmTolerance);
}

TEST(GemmAvx2Test, MatchesScalarAcrossBlockBoundaries) {
    Tensor a = Tensor::random(Shape({130, 300}), -1.0f, 1.0f);
    Tensor b = Tensor::random(Shape({300, 270}), -1.0f, 1.0f);
    ExpectTensorsClose(matmul_scalar(a, b), matmul_avx2(a, b), 1e-3f);
}

TEST(GemmAvx2Test, MatchesScalarOnNonSquareTallSkinny) {
    Tensor a = Tensor::random(Shape({500, 16}), -1.0f, 1.0f);
    Tensor b = Tensor::random(Shape({16, 4}), -1.0f, 1.0f);
    ExpectTensorsClose(matmul_scalar(a, b), matmul_avx2(a, b), kGemmTolerance);
}

TEST(ReluAvx2Test, MatchesScalarExactly) {
    Tensor t = Tensor::random(Shape({137}), -5.0f, 5.0f);
    Tensor expected = t.clone(), actual = t.clone();
    relu_scalar(expected);
    relu_avx2(actual);
    ExpectTensorsClose(expected, actual, 0.0f);
}

TEST(ReluAvx2Test, MatchesScalarOnExactMultipleOf8) {
    Tensor t = Tensor::random(Shape({64}), -5.0f, 5.0f);
    Tensor expected = t.clone(), actual = t.clone();
    relu_scalar(expected);
    relu_avx2(actual);
    ExpectTensorsClose(expected, actual, 0.0f);
}
#endif

TEST(GemmDispatchTest, MatmulMatchesScalarRegardlessOfBackend) {
    Tensor a = Tensor::random(Shape({16, 24}));
    Tensor b = Tensor::random(Shape({24, 12}));
    ExpectTensorsClose(matmul_scalar(a, b), matmul(a, b), kGemmTolerance);
}

TEST(ReluDispatchTest, ReluMatchesScalarRegardlessOfBackend) {
    Tensor t = Tensor::random(Shape({50}), -3.0f, 3.0f);
    Tensor expected = t.clone();
    relu_scalar(expected);
    Tensor actual = relu(t);
    ExpectTensorsClose(expected, actual, 0.0f);
}