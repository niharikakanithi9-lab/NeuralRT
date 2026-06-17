#pragma once
#include <vector>

#include "neuralrt/graph/graph.h"

namespace neuralrt::graph {

struct TensorLifetime {
    TensorId tensor_id;
    size_t produced_at_step;
    size_t last_used_at_step;
};

struct ExecutionPlan {
    std::vector<NodeId> execution_order;
    std::vector<size_t> tensor_to_slot;
    size_t num_buffer_slots = 0;
};

class Planner {
public:
    static ExecutionPlan plan(const Graph& graph);
};

}  // namespace neuralrt::graph