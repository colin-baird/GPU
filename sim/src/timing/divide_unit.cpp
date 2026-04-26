#include "gpu_sim/timing/divide_unit.h"

namespace gpu_sim {

void DivideUnit::accept(const DispatchInput& input, uint64_t cycle) {
    // Phase 1 discipline: writes only into next_* slots.
    next_busy_ = true;
    next_cycles_remaining_ = DIVIDE_LATENCY;
    next_pending_result_.valid = true;
    next_pending_result_.warp_id = input.warp_id;
    next_pending_result_.dest_reg = input.decoded.rd;
    next_pending_result_.values = input.trace.results;
    next_pending_result_.source_unit = ExecUnit::DIVIDE;
    next_pending_result_.pc = input.pc;
    next_pending_result_.raw_instruction = input.decoded.raw;
    next_pending_result_.issue_cycle = cycle;
    stats_.div_stats.instructions++;
}

void DivideUnit::evaluate() {
    // Operates on next_* (which was seeded equal to current_* by commit()
    // at the end of the prior tick, and may have been updated by accept()
    // earlier in this tick).
    if (next_busy_) {
        stats_.div_stats.busy_cycles++;
        next_cycles_remaining_--;
        if (next_cycles_remaining_ == 0) {
            next_result_buffer_ = next_pending_result_;
            next_busy_ = false;
        }
    }
}

void DivideUnit::commit() {
    current_busy_ = next_busy_;
    current_cycles_remaining_ = next_cycles_remaining_;
    current_pending_result_ = next_pending_result_;
    current_result_buffer_ = next_result_buffer_;
}

void DivideUnit::reset() {
    current_busy_ = false;
    next_busy_ = false;
    current_cycles_remaining_ = 0;
    next_cycles_remaining_ = 0;
    current_result_buffer_.valid = false;
    next_result_buffer_.valid = false;
    current_pending_result_.valid = false;
    next_pending_result_.valid = false;
}

bool DivideUnit::has_result() const {
    // COMBINATIONAL edge with the writeback arbiter (same-tick visibility,
    // zero cycle delta).
    return next_result_buffer_.valid;
}

WritebackEntry DivideUnit::consume_result() {
    WritebackEntry entry = next_result_buffer_;
    next_result_buffer_.valid = false;
    return entry;
}

} // namespace gpu_sim
