#pragma once
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "neuralrt/core/tensor.h"

namespace neuralrt::graph {

using NodeId = size_t;
using TensorId = size_t;

enum class OpType { kInput, kLinear, kReLU };

struct Node {
    NodeId id;
    OpType op;
    std::vector<TensorId> inputs;
    TensorId output;
    std::string name;
    size_t linear_in_features = 0;
    size_t linear_out_features = 0;
};

class Graph {
public:
    Graph() = default;

    TensorId add_input(const std::string& name, Shape shape);
    NodeId add_linear(const std::string& name, TensorId input, size_t in_features,
                       size_t out_features);
    NodeId add_relu(const std::string& name, TensorId input);

    void mark_output(TensorId id);

    const std::vector<Node>& nodes() const noexcept;
    const std::unordered_map<TensorId, Shape>& input_shapes() const noexcept;
    const std::vector<TensorId>& outputs() const noexcept;
    const std::vector<TensorId>& inputs() const noexcept;

    size_t num_tensors() const noexcept;

private:
    TensorId next_tensor_id_ = 0;
    NodeId next_node_id_ = 0;

    std::vector<Node> nodes_;
    std::unordered_map<TensorId, Shape> input_shapes_;
    std::vector<TensorId> inputs_;
    std::vector<TensorId> outputs_;

    TensorId new_tensor_id();
};

}  // namespace neuralrt::graph