#pragma once
#include <unordered_map>
#include <vector>

#include "neuralrt/graph/graph.h"
#include "neuralrt/graph/planner.h"
#include "neuralrt/ops/linear.h"

namespace neuralrt::graph {

class Executor {
public:
    Executor(Graph graph, size_t max_batch_size);

    const Tensor& run(const Tensor& input);

private:
    Graph graph_;
    ExecutionPlan plan_;
    size_t max_batch_size_;

    std::unordered_map<NodeId, ops::Linear> linear_layers_;
    std::vector<Tensor> slot_buffers_;
    std::vector<Tensor> live_views_;
    Tensor result_;
    std::unordered_map<TensorId, size_t> node_feature_dim_;

    TensorId graph_input_id_;
    TensorId graph_output_id_;

    void build_layers();
    void allocate_buffers();
};

}  // namespace neuralrt::graph