#include <benchmark/benchmark.h>
#include "neuralrt/core/tensor.h"
#include "neuralrt/model/sequential.h"

using neuralrt::Shape;
using neuralrt::Tensor;
using neuralrt::model::LinearLayer;
using neuralrt::model::ReLULayer;
using neuralrt::model::Sequential;

static void BM_MlpForwardBatch1(benchmark::State& state) {
    Sequential model;
    model.add(std::make_unique<LinearLayer>(128, 256));
    model.add(std::make_unique<ReLULayer>());
    model.add(std::make_unique<LinearLayer>(256, 64));
    model.add(std::make_unique<ReLULayer>());
    model.add(std::make_unique<LinearLayer>(64, 10));
    Tensor input = Tensor::random(Shape({1, 128}));
    for (auto _ : state) {
        Tensor out = model.forward(input);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_MlpForwardBatch1);

static void BM_MlpForwardBatched(benchmark::State& state) {
    const size_t batch = static_cast<size_t>(state.range(0));
    Sequential model;
    model.add(std::make_unique<LinearLayer>(128, 256));
    model.add(std::make_unique<ReLULayer>());
    model.add(std::make_unique<LinearLayer>(256, 64));
    model.add(std::make_unique<ReLULayer>());
    model.add(std::make_unique<LinearLayer>(64, 10));
    Tensor input = Tensor::random(Shape({batch, 128}));
    for (auto _ : state) {
        Tensor out = model.forward(input);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_MlpForwardBatched)->Arg(1)->Arg(8)->Arg(32)->Arg(128);