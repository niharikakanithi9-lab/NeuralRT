#include "neuralrt/graph/graph.h"

#include <gtest/gtest.h>
#include <stdexcept>

#include "neuralrt/graph/executor.h"
#include "neuralrt/graph/planner.h"

using namespace neuralrt;
using namespace neuralrt::graph;

namespace {
Graph build_mlp_graph(size_t in_features, size_t hidden, size_t out_features) {
    Graph g;
    TensorId input = g.add_input("input", Shape({1, in_features}));
    NodeId fc1 = g.add_linear("fc1", input, in_features, hidden);
    NodeId relu1 = g.add_relu("relu1", g.nodes()[fc1].output);
    NodeId fc2 = g.add_linear("fc2", g.nodes()[relu1].output, hidden, out_features);
    g.mark_output(g.nodes()[fc2].output);
    return g;
}
}  // namespace

TEST(GraphTest, BuildsExpectedNodeCount) {
    Graph g = build_mlp_graph(4, 8, 2);
    EXPECT_EQ(g.nodes().size(), 4u);
}

TEST(GraphTest, InputsAndOutputsTracked) {
    Graph g = build_mlp_graph(4, 8, 2);
    EXPECT_EQ(g.inputs().size(), 1u);
    EXPECT_EQ(g.outputs().size(), 1u);
}

TEST(PlannerTest, ProducesValidTopologicalOrder) {
    Graph g = build_mlp_graph(4, 8, 2);
    ExecutionPlan plan = Planner::plan(g);
    EXPECT_EQ(plan.execution_order.size(), g.nodes().size());

    std::unordered_map<NodeId, size_t> position;
    for (size_t i = 0; i < plan.execution_order.size(); ++i) position[plan.execution_order[i]] = i;
    std::unordered_map<TensorId, NodeId> producer;
    for (const Node& n : g.nodes()) producer[n.output] = n.id;

    for (const Node& n : g.nodes()) {
        for (TensorId in_tensor : n.inputs) {
            NodeId dep_node = producer.at(in_tensor);
            EXPECT_LT(position.at(dep_node), position.at(n.id));
        }
    }
}

TEST(PlannerTest, ReusesBufferSlots) {
    Graph g = build_mlp_graph(4, 8, 2);
    ExecutionPlan plan = Planner::plan(g);
    EXPECT_LT(plan.num_buffer_slots, g.num_tensors());
}

TEST(ExecutorTest, ProducesCorrectOutputShape) {
    Graph g = build_mlp_graph(4, 8, 2);
    Executor exec(std::move(g), 16);
    Tensor input = Tensor::random(Shape({3, 4}));
    const Tensor& output = exec.run(input);
    EXPECT_EQ(output.shape().dim(0), 3u);
    EXPECT_EQ(output.shape().dim(1), 2u);
}

TEST(ExecutorTest, ThrowsWhenBatchExceedsMax) {
    Graph g = build_mlp_graph(4, 8, 2);
    Executor exec(std::move(g), 2);
    Tensor input = Tensor::random(Shape({5, 4}));
    EXPECT_THROW(exec.run(input), std::invalid_argument);
}

TEST(ExecutorTest, HandlesVaryingBatchSizesWithinMax) {
    Graph g = build_mlp_graph(4, 8, 2);
    Executor exec(std::move(g), 16);
    Tensor small = Tensor::random(Shape({1, 4}));
    Tensor large = Tensor::random(Shape({16, 4}));
    const Tensor& out_small = exec.run(small);
    EXPECT_EQ(out_small.shape().dim(0), 1u);
    const Tensor& out_large = exec.run(large);
    EXPECT_EQ(out_large.shape().dim(0), 16u);
}

TEST(ExecutorTest, MatchesManualLinearReluLinearComputation) {
    Graph g;
    TensorId input = g.add_input("input", Shape({1, 2}));
    NodeId fc1 = g.add_linear("fc1", input, 2, 2);
    NodeId relu1 = g.add_relu("relu1", g.nodes()[fc1].output);
    g.mark_output(g.nodes()[relu1].output);

    Executor exec(std::move(g), 4);
    Tensor in(Shape({1, 2}), {1.0f, 2.0f});
    const Tensor& out = exec.run(in);
    for (size_t i = 0; i < out.size(); ++i) EXPECT_GE(out.data()[i], 0.0f);
}