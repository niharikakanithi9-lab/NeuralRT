#include <benchmark/benchmark.h>

#include "neuralrt/core/tensor.h"
#include "neuralrt/graph/executor.h"
#include "neuralrt/graph/graph.h"
#include "neuralrt/model/sequential.h"

using namespace neuralrt;
using namespace neuralrt::graph;
using neuralrt::model::LinearLayer;
using neuralrt::model::ReLULayer;
using neuralrt::model::Sequential;

namespace {
Graph build_mlp_graph() {
    Graph g;
    TensorId input = g.add_input("input", Shape({1, 128}));
    NodeId fc1 = g.add_linear("fc1", input, 128, 256);
    NodeId relu1 = g.add_relu("relu1", g.nodes()[fc1].output);
    NodeId fc2 = g.add_linear("fc2", g.nodes()[relu1].output, 256, 64);
    NodeId relu2 = g.add_relu("relu2", g.nodes()[fc2].output);
    NodeId fc3 = g.add_linear("fc3", g.nodes()[relu2].output, 64, 10);
    g.mark_output(g.nodes()[fc3].output);
    return g;
}
}  // namespace

static void BM_SequentialForwardBatch1(benchmark::State& state) {
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
BENCHMARK(BM_SequentialForwardBatch1);

static void BM_ExecutorRunBatch1(benchmark::State& state) {
    Executor exec(build_mlp_graph(), 1);
    Tensor input = Tensor::random(Shape({1, 128}));
    for (auto _ : state) {
        const Tensor& out = exec.run(input);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_ExecutorRunBatch1);

static void BM_ExecutorRunBatched(benchmark::State& state) {
    const size_t batch = static_cast<size_t>(state.range(0));
    Executor exec(build_mlp_graph(), batch);
    Tensor input = Tensor::random(Shape({batch, 128}));
    for (auto _ : state) {
        const Tensor& out = exec.run(input);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_ExecutorRunBatched)->Arg(1)->Arg(8)->Arg(32)->Arg(128);