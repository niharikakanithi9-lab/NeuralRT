#include <benchmark/benchmark.h>
#include "neuralrt/core/tensor.h"
#include "neuralrt/ops/activation.h"

using neuralrt::Shape;
using neuralrt::Tensor;
using neuralrt::ops::relu_scalar;

static void BM_ReluScalar(benchmark::State& state) {
    const size_t n = static_cast<size_t>(state.range(0));
    Tensor t = Tensor::random(Shape({n}), -5.0f, 5.0f);
    for (auto _ : state) {
        relu_scalar(t);
        benchmark::DoNotOptimize(t.data());
    }
}
BENCHMARK(BM_ReluScalar)->Arg(1024)->Arg(65536)->Arg(1048576);

#if defined(__AVX2__)
using neuralrt::ops::relu_avx2;

static void BM_ReluAvx2(benchmark::State& state) {
    const size_t n = static_cast<size_t>(state.range(0));
    Tensor t = Tensor::random(Shape({n}), -5.0f, 5.0f);
    for (auto _ : state) {
        relu_avx2(t);
        benchmark::DoNotOptimize(t.data());
    }
}
BENCHMARK(BM_ReluAvx2)->Arg(1024)->Arg(65536)->Arg(1048576);
#endif