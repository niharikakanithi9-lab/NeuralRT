#include "neuralrt/graph/executor.h"
#include <cstdint>
#include <algorithm>
#include <stdexcept>

#include "neuralrt/ops/activation.h"

namespace neuralrt::graph {

Executor::Executor(Graph graph, size_t max_batch_size)
    : graph_(std::move(graph)), max_batch_size_(max_batch_size) {
    if (graph_.inputs().size() != 1) throw std::invalid_argument("Executor: only single-input graphs are supported");
    if (graph_.outputs().size() != 1) throw std::invalid_argument("Executor: only single-output graphs are supported");
    graph_input_id_ = graph_.inputs().front();
    graph_output_id_ = graph_.outputs().front();

    plan_ = Planner::plan(graph_);
    build_layers();
    allocate_buffers();
    live_views_.resize(graph_.num_tensors());
}

void Executor::build_layers() {
    for (const Node& node : graph_.nodes()) {
        if (node.op == OpType::kLinear) {
            linear_layers_.emplace(node.id, ops::Linear(node.linear_in_features, node.linear_out_features));
        }
    }
}

void Executor::allocate_buffers() {
    auto& feature_dim = node_feature_dim_;

    const auto& input_shapes = graph_.input_shapes();
    auto it = input_shapes.find(graph_input_id_);
    if (it == input_shapes.end()) throw std::runtime_error("Executor: graph input has no declared shape");
    feature_dim[graph_input_id_] = it->second.dim(1);

    for (NodeId node_id : plan_.execution_order) {
        const Node& node = graph_.nodes()[node_id];
        if (node.op == OpType::kInput) continue;
        if (node.op == OpType::kLinear) {
            feature_dim[node.output] = node.linear_out_features;
        } else if (node.op == OpType::kReLU) {
            feature_dim[node.output] = feature_dim.at(node.inputs[0]);
        }
    }

    std::vector<size_t> slot_feature_dim(plan_.num_buffer_slots, 0);
    for (size_t i = 0; i < plan_.tensor_to_slot.size(); ++i) {
        size_t slot = plan_.tensor_to_slot[i];
        if (slot == SIZE_MAX) continue;
        size_t feat = feature_dim.count(i) ? feature_dim.at(i) : 0;
        slot_feature_dim[slot] = std::max(slot_feature_dim[slot], feat);
    }

    slot_buffers_.clear();
    slot_buffers_.reserve(plan_.num_buffer_slots);
    for (size_t feat : slot_feature_dim) slot_buffers_.emplace_back(Tensor::zeros(Shape({max_batch_size_, feat})));
}

const Tensor& Executor::run(const Tensor& input) {
    const size_t batch = input.shape().dim(0);
    if (batch > max_batch_size_) throw std::invalid_argument("Executor::run: batch size exceeds max_batch_size set at construction");

    const auto& input_shapes = graph_.input_shapes();
    const size_t declared_input_features = input_shapes.at(graph_input_id_).dim(1);

    for (NodeId node_id : plan_.execution_order) {
        const Node& node = graph_.nodes()[node_id];
        size_t slot = plan_.tensor_to_slot[node.output];
        Tensor& slot_buf = slot_buffers_[slot];

        if (node.op == OpType::kInput) {
            if (node.output != graph_input_id_) throw std::runtime_error("Executor: unexpected multiple input nodes");
            if (input.shape().dim(1) != declared_input_features) throw std::invalid_argument("Executor::run: input feature dim mismatch");
            float* dst = slot_buf.data();
            const float* src = input.data();
            std::copy(src, src + batch * declared_input_features, dst);
            live_views_[node.output] = Tensor::view(Shape({batch, declared_input_features}), slot_buf.data());
            continue;
        }

        const size_t feat = node_feature_dim_.at(node.output);
        Tensor out_view = Tensor::view(Shape({batch, feat}), slot_buf.data());

        if (node.op == OpType::kLinear) {
            const Tensor& in_view = live_views_.at(node.inputs[0]);
            linear_layers_.at(node.id).forward_into(in_view, out_view);
        } else if (node.op == OpType::kReLU) {
            const Tensor& in_view = live_views_.at(node.inputs[0]);
            std::copy(in_view.data(), in_view.data() + in_view.size(), out_view.data());
            ops::relu_inplace(out_view);
        }

        live_views_[node.output] = out_view;
    }

    result_ = live_views_.at(graph_output_id_);
    return result_;
}

}  // namespace neuralrt::graph