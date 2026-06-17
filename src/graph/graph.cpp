#include "neuralrt/graph/graph.h"

namespace neuralrt::graph {

TensorId Graph::new_tensor_id() { return next_tensor_id_++; }

TensorId Graph::add_input(const std::string& name, Shape shape) {
    TensorId id = new_tensor_id();
    input_shapes_[id] = std::move(shape);
    inputs_.push_back(id);

    Node node;
    node.id = next_node_id_++;
    node.op = OpType::kInput;
    node.output = id;
    node.name = name;
    nodes_.push_back(std::move(node));
    return id;
}

NodeId Graph::add_linear(const std::string& name, TensorId input, size_t in_features,
                          size_t out_features) {
    TensorId out_id = new_tensor_id();

    Node node;
    node.id = next_node_id_++;
    node.op = OpType::kLinear;
    node.inputs = {input};
    node.output = out_id;
    node.name = name;
    node.linear_in_features = in_features;
    node.linear_out_features = out_features;

    NodeId id = node.id;
    nodes_.push_back(std::move(node));
    return id;
}

NodeId Graph::add_relu(const std::string& name, TensorId input) {
    TensorId out_id = new_tensor_id();

    Node node;
    node.id = next_node_id_++;
    node.op = OpType::kReLU;
    node.inputs = {input};
    node.output = out_id;
    node.name = name;

    NodeId id = node.id;
    nodes_.push_back(std::move(node));
    return id;
}

void Graph::mark_output(TensorId id) { outputs_.push_back(id); }

const std::vector<Node>& Graph::nodes() const noexcept { return nodes_; }
const std::unordered_map<TensorId, Shape>& Graph::input_shapes() const noexcept { return input_shapes_; }
const std::vector<TensorId>& Graph::outputs() const noexcept { return outputs_; }
const std::vector<TensorId>& Graph::inputs() const noexcept { return inputs_; }
size_t Graph::num_tensors() const noexcept { return next_tensor_id_; }

}  // namespace neuralrt::graph