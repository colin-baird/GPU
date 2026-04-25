#include "gpu_sim/timing/tlookup_unit.h"

namespace gpu_sim {

void TLookupUnit::accept(const DispatchInput& input, uint64_t cycle) {
    // Phase 1 discipline: writes only into next_* slots.
    next_busy_ = true;
    next_cycles_remaining_ = TLOOKUP_LATENCY;
    next_pending_result_.valid = true;
    next_pending_result_.warp_id = input.warp_id;
    next_pending_result_.dest_reg = input.decoded.rd;
    next_pending_result_.values = input.trace.results;
    next_pending_result_.source_unit = ExecUnit::TLOOKUP;
    next_pending_result_.pc = input.pc;
    next_pending_result_.raw_instruction = input.decoded.raw;
    next_pending_result_.issue_cycle = cycle;
    stats_.tlookup_stats.instructions++;
}

void TLookupUnit::evaluate() {
    // Operates on next_* (seeded equal to current_* by commit() at the end
    // of the prior tick, possibly updated by accept() earlier this tick).
    if (next_busy_) {
        stats_.tlookup_stats.busy_cycles++;
        next_cycles_remaining_--;
        if (next_cycles_remaining_ == 0) {
            next_result_buffer_ = next_pending_result_;
            next_busy_ = false;
        }
    }
}

void TLookupUnit::commit() {
    current_busy_ = next_busy_;
    current_cycles_remaining_ = next_cycles_remaining_;
    current_pending_result_ = next_pending_result_;
    current_result_buffer_ = next_result_buffer_;
}

void TLookupUnit::reset() {
    current_busy_ = false;
    next_busy_ = false;
    current_cycles_remaining_ = 0;
    next_cycles_remaining_ = 0;
    current_result_buffer_.valid = false;
    next_result_buffer_.valid = false;
    current_pending_result_.valid = false;
    next_pending_result_.valid = false;
}

bool TLookupUnit::is_ready() const {
    // Queried by scheduler before this tick's evaluate; reads committed
    // end-of-last-cycle state.
    return !current_busy_ && !current_result_buffer_.valid;
}

bool TLookupUnit::has_result() const {
    // COMBINATIONAL edge with the writeback arbiter (same-tick visibility,
    // zero cycle delta).
    return next_result_buffer_.valid;
}

WritebackEntry TLookupUnit::consume_result() {
    WritebackEntry entry = next_result_buffer_;
    next_result_buffer_.valid = false;
    return entry;
}

} // namespace gpu_sim
