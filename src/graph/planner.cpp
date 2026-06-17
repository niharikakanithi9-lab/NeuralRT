#include "neuralrt/graph/planner.h"
#include <cstdint>
#include <algorithm>
#include <queue>
#include <stdexcept>
#include <unordered_map>

namespace neuralrt::graph {

namespace {
std::vector<size_t> topological_order(const Graph& g) {
    const auto& nodes = g.nodes();
    const size_t n = nodes.size();

    std::unordered_map<TensorId, size_t> producer_of;
    for (size_t i = 0; i < n; ++i) producer_of[nodes[i].output] = i;

    std::vector<size_t> in_degree(n, 0);
    std::vector<std::vector<size_t>> dependents(n);

    for (size_t i = 0; i < n; ++i) {
        for (TensorId in_tensor : nodes[i].inputs) {
            auto it = producer_of.find(in_tensor);
            if (it == producer_of.end()) throw std::runtime_error("Planner: unknown tensor id");
            dependents[it->second].push_back(i);
            ++in_degree[i];
        }
    }

    std::queue<size_t> ready;
    for (size_t i = 0; i < n; ++i) if (in_degree[i] == 0) ready.push(i);

    std::vector<size_t> order;
    order.reserve(n);
    while (!ready.empty()) {
        size_t cur = ready.front();
        ready.pop();
        order.push_back(cur);
        for (size_t dep : dependents[cur]) if (--in_degree[dep] == 0) ready.push(dep);
    }

    if (order.size() != n) throw std::runtime_error("Planner: graph contains a cycle");
    return order;
}
}  // namespace

ExecutionPlan Planner::plan(const Graph& graph) {
    ExecutionPlan plan;
    const auto& nodes = graph.nodes();

    std::vector<size_t> order_idx = topological_order(graph);
    plan.execution_order.reserve(order_idx.size());
    for (size_t idx : order_idx) plan.execution_order.push_back(nodes[idx].id);

    const size_t num_tensors = graph.num_tensors();
    std::vector<size_t> last_used_at(num_tensors, SIZE_MAX);

    for (size_t step = 0; step < order_idx.size(); ++step) {
        const Node& node = nodes[order_idx[step]];
        for (TensorId in_tensor : node.inputs) last_used_at[in_tensor] = step;
    }
    for (TensorId out_id : graph.outputs()) last_used_at[out_id] = SIZE_MAX;

    plan.tensor_to_slot.assign(num_tensors, SIZE_MAX);
    std::vector<TensorId> slot_owner;
    std::vector<bool> slot_free;

    auto find_or_create_free_slot = [&]() -> size_t {
        for (size_t s = 0; s < slot_free.size(); ++s) if (slot_free[s]) return s;
        slot_free.push_back(false);
        slot_owner.push_back(SIZE_MAX);
        return slot_free.size() - 1;
    };

    for (size_t step = 0; step < order_idx.size(); ++step) {
        for (size_t s = 0; s < slot_owner.size(); ++s) {
            if (!slot_free[s] && slot_owner[s] != SIZE_MAX) {
                TensorId occupant = slot_owner[s];
                if (last_used_at[occupant] != SIZE_MAX && last_used_at[occupant] < step) {
                    slot_free[s] = true;
                    slot_owner[s] = SIZE_MAX;
                }
            }
        }
        const Node& node = nodes[order_idx[step]];
        size_t slot = find_or_create_free_slot();
        slot_free[slot] = false;
        slot_owner[slot] = node.output;
        plan.tensor_to_slot[node.output] = slot;
    }

    plan.num_buffer_slots = slot_free.size();
    return plan;
}

}  // namespace neuralrt::graph