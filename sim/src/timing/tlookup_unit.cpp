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
    // Phase 10B.0.5: tlookup_stats.instructions is incremented in commit()
    // (gated on accepted_this_cycle_), not here.
    accepted_this_cycle_ = true;
}

void TLookupUnit::seed_next() {
    // Phase 10B.0.5: re-establish the carry-forward iterative state in next_*
    // at the top of the tick.
    next_busy_ = current_busy_;
    next_cycles_remaining_ = current_cycles_remaining_;
    next_pending_result_ = current_pending_result_;
}

void TLookupUnit::evaluate() {
    // Operates on next_* (seeded equal to current_* by seed_next() at the top
    // of the tick, possibly updated by accept() earlier this tick).
    // Phase 10B.0.5: capture the per-cycle busy flag before the body may
    // clear next_busy_; tlookup_stats.busy_cycles is incremented at commit().
    busy_this_cycle_ = next_busy_;
    if (next_busy_) {
        next_cycles_remaining_--;
        if (next_cycles_remaining_ == 0) {
            next_result_buffer_ = next_pending_result_;
            next_busy_ = false;
        }
    }
}

void TLookupUnit::commit() {
    // Phase 10B.0.5: Stats increments relocated here from evaluate()/accept().
    // Both per-cycle flags are consumed and cleared at commit() so a commit()
    // not preceded by an evaluate() never re-counts a stale flag.
    if (busy_this_cycle_) {
        stats_.tlookup_stats.busy_cycles++;
        busy_this_cycle_ = false;
    }
    if (accepted_this_cycle_) {
        stats_.tlookup_stats.instructions++;
        accepted_this_cycle_ = false;
    }

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
    busy_this_cycle_ = false;
    accepted_this_cycle_ = false;
}

bool TLookupUnit::next_has_result() const {
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
