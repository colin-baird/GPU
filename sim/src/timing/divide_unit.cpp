#include "gpu_sim/timing/divide_unit.h"
#include "gpu_sim/timing/writeback_arbiter.h"

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
    // Phase 10B.0.5: div_stats.instructions is incremented in commit()
    // (gated on accepted_this_cycle_), not here.
    accepted_this_cycle_ = true;
}

void DivideUnit::seed_next() {
    // Phase 10B.0.5: re-establish the carry-forward iterative state in next_*
    // at the top of the tick.
    next_busy_ = current_busy_;
    next_cycles_remaining_ = current_cycles_remaining_;
    next_pending_result_ = current_pending_result_;
}

void DivideUnit::evaluate() {
    // Phase 10B.1: REGISTERED opcoll->unit edge via the pull model.
    if (opcoll_ != nullptr) {
        const auto& dispatched = opcoll_->current_output();
        if (dispatched && dispatched->decoded.target_unit == ExecUnit::DIVIDE) {
            accept(*dispatched, sim_cycle_ != nullptr ? *sim_cycle_ : 0);
        }
    }

    // Operates on next_* (which was seeded equal to current_* by seed_next()
    // at the top of the tick, and may have been updated by accept() earlier
    // in this tick).
    // Phase 10B.3: the result buffer is a plain double-buffered pipeline
    // register — assign next_result_buffer_ fresh every cycle.
    next_result_buffer_ = WritebackEntry{};
    // Phase 10B.0.5: capture the per-cycle busy flag before the body may
    // clear next_busy_; div_stats.busy_cycles is incremented at commit().
    busy_this_cycle_ = next_busy_;
    if (next_busy_) {
        next_cycles_remaining_--;
        if (next_cycles_remaining_ == 0) {
            next_result_buffer_ = next_pending_result_;
            next_busy_ = false;
        }
    }
}

void DivideUnit::commit() {
    // Phase 10B.3: writeback-stall self-gate. On a stalled cycle this stage
    // holds — current_busy_/cycles_remaining_/pending_result_/result_buffer_
    // all hold, so the next tick's seed_next()+evaluate() re-derive the
    // identical countdown step and (if the countdown reached 0) the identical
    // result. The countdown is decremented at most once per non-stalled cycle.
    if (wb_arbiter_ != nullptr && wb_arbiter_->next_writeback_stall()) {
        return;
    }

    // Phase 10B.0.5: Stats increments relocated here from evaluate()/accept().
    // Both per-cycle flags are consumed and cleared at commit() so a commit()
    // not preceded by an evaluate() never re-counts a stale flag.
    if (busy_this_cycle_) {
        stats_.div_stats.busy_cycles++;
        busy_this_cycle_ = false;
    }
    if (accepted_this_cycle_) {
        stats_.div_stats.instructions++;
        accepted_this_cycle_ = false;
    }

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
    busy_this_cycle_ = false;
    accepted_this_cycle_ = false;
}

bool DivideUnit::current_has_result() const {
    // Phase 10B.3: REGISTERED unit->arbiter edge — committed (current_*) state.
    return current_result_buffer_.valid;
}

WritebackEntry DivideUnit::consume_result() {
    // Phase 10B.3: pure read — mutates nothing.
    return current_result_buffer_;
}

} // namespace gpu_sim
