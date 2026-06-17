#include <benchmark/benchmark.h>
#include "neuralrt/core/tensor.h"
#include "neuralrt/ops/gemm.h"

using neuralrt::Shape;
using neuralrt::Tensor;
using neuralrt::ops::matmul_scalar;

static void BM_MatmulScalar(benchmark::State& state) {
    const size_t dim = static_cast<size_t>(state.range(0));
    Tensor a = Tensor::random(Shape({dim, dim}));
    Tensor b = Tensor::random(Shape({dim, dim}));
    for (auto _ : state) {
        Tensor c = matmul_scalar(a, b);
        benchmark::DoNotOptimize(c.data());
    }
    state.SetComplexityN(static_cast<int64_t>(dim));
}
BENCHMARK(BM_MatmulScalar)->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Arg(512);

#if defined(__AVX2__)
using neuralrt::ops::matmul_avx2;

static void BM_MatmulAvx2(benchmark::State& state) {
    const size_t dim = static_cast<size_t>(state.range(0));
    Tensor a = Tensor::random(Shape({dim, dim}));
    Tensor b = Tensor::random(Shape({dim, dim}));
    for (auto _ : state) {
        Tensor c = matmul_avx2(a, b);
        benchmark::DoNotOptimize(c.data());
    }
    state.SetComplexityN(static_cast<int64_t>(dim));
}
BENCHMARK(BM_MatmulAvx2)->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Arg(512);
#endif

BENCHMARK_MAIN();