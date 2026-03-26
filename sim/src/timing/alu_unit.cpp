#include "gpu_sim/timing/alu_unit.h"

namespace gpu_sim {

void ALUUnit::accept(const DispatchInput& input, uint64_t cycle) {
    has_pending_ = true;
    pending_input_ = input;
    pending_cycle_ = cycle;
}

void ALUUnit::evaluate() {
    if (has_pending_) {
        // 1-cycle: result available immediately
        result_buffer_.valid = true;
        result_buffer_.warp_id = pending_input_.warp_id;
        result_buffer_.dest_reg = pending_input_.decoded.rd;
        result_buffer_.values = pending_input_.trace.results;
        result_buffer_.source_unit = ExecUnit::ALU;
        result_buffer_.issue_cycle = pending_cycle_;
        has_pending_ = false;
        stats_.alu_stats.busy_cycles++;
        stats_.alu_stats.instructions++;
    }
}

void ALUUnit::commit() {
    // No double-buffering needed for result buffer -- it's consumed by writeback arbiter
}

void ALUUnit::reset() {
    result_buffer_.valid = false;
    has_pending_ = false;
}

bool ALUUnit::is_ready() const {
    return !result_buffer_.valid && !has_pending_;
}

bool ALUUnit::has_result() const {
    return result_buffer_.valid;
}

WritebackEntry ALUUnit::consume_result() {
    WritebackEntry entry = result_buffer_;
    result_buffer_.valid = false;
    return entry;
}

} // namespace gpu_sim
