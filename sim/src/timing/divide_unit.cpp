#include "gpu_sim/timing/divide_unit.h"

namespace gpu_sim {

void DivideUnit::accept(const DispatchInput& input, uint64_t cycle) {
    busy_ = true;
    cycles_remaining_ = DIVIDE_LATENCY;
    pending_result_.valid = true;
    pending_result_.warp_id = input.warp_id;
    pending_result_.dest_reg = input.decoded.rd;
    pending_result_.values = input.trace.results;
    pending_result_.source_unit = ExecUnit::DIVIDE;
    pending_result_.issue_cycle = cycle;
    stats_.div_stats.instructions++;
}

void DivideUnit::evaluate() {
    if (busy_) {
        stats_.div_stats.busy_cycles++;
        cycles_remaining_--;
        if (cycles_remaining_ == 0) {
            result_buffer_ = pending_result_;
            busy_ = false;
        }
    }
}

void DivideUnit::commit() {}

void DivideUnit::reset() {
    busy_ = false;
    cycles_remaining_ = 0;
    result_buffer_.valid = false;
    pending_result_.valid = false;
}

bool DivideUnit::is_ready() const {
    return !busy_ && !result_buffer_.valid;
}

bool DivideUnit::has_result() const {
    return result_buffer_.valid;
}

WritebackEntry DivideUnit::consume_result() {
    WritebackEntry entry = result_buffer_;
    result_buffer_.valid = false;
    return entry;
}

} // namespace gpu_sim
