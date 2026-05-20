#include "gpu_sim/timing/tlookup_unit.h"
#include "gpu_sim/timing/writeback_arbiter.h"

namespace gpu_sim {

void TLookupUnit::accept(const DispatchInput& input, uint64_t cycle) {
    // Phase 1 discipline: writes only into the staged slot.
    busy_.set_next(true);
    cycles_remaining_.set_next(TLOOKUP_LATENCY);
    WritebackEntry& pr = pending_result_.next_mut();
    pr.valid = true;
    pr.warp_id = input.warp_id;
    pr.dest_reg = input.decoded.rd;
    pr.values = input.trace.results;
    pr.source_unit = ExecUnit::TLOOKUP;
    pr.pc = input.pc;
    pr.raw_instruction = input.decoded.raw;
    pr.issue_cycle = cycle;
    // Phase 10B.0.5: tlookup_stats.instructions is incremented in commit()
    // (gated on accepted_this_cycle_), not here.
    accepted_this_cycle_.drive(true);
}

void TLookupUnit::evaluate() {
    // Phase 10B.1: REGISTERED opcoll->unit edge via the pull model.
    if (opcoll_ != nullptr) {
        const auto& dispatched = opcoll_->current_output();
        if (dispatched && dispatched->decoded.target_unit == ExecUnit::TLOOKUP) {
            accept(*dispatched, sim_cycle_ != nullptr ? *sim_cycle_ : 0);
        }
    }

    // Operates on the staged slots (seeded equal to current_* by seed_next()
    // at the top of the tick, possibly updated by accept() earlier this tick).
    // Phase 10B.3: the result buffer is a plain double-buffered pipeline
    // register — assign it fresh every cycle.
    result_buffer_.set_next(WritebackEntry{});
    // Phase 10B.0.5: capture the per-cycle busy flag before the body may
    // clear next_busy_; tlookup_stats.busy_cycles is incremented at commit().
    busy_this_cycle_.drive(busy_.next());
    if (busy_.next()) {
        cycles_remaining_.next_mut()--;
        if (cycles_remaining_.next() == 0) {
            result_buffer_.set_next(pending_result_.next());
            busy_.set_next(false);
        }
    }
}

void TLookupUnit::commit() {
    // Phase 10B.3: writeback-stall self-gate. On a stalled cycle this stage
    // holds — the countdown is decremented at most once per non-stalled cycle.
    if (wb_arbiter_ != nullptr && wb_arbiter_->next_writeback_stall()) {
        return;
    }

    // Phase 10B.0.5: Stats increments relocated here from evaluate()/accept().
    // Both per-cycle flags are consumed and cleared at commit() so a commit()
    // not preceded by an evaluate() never re-counts a stale flag.
    if (busy_this_cycle_.value()) {
        stats_.tlookup_stats.busy_cycles++;
    }
    if (accepted_this_cycle_.value()) {
        stats_.tlookup_stats.instructions++;
    }
    busy_this_cycle_.reset();
    accepted_this_cycle_.reset();

    commit_all();
}

bool TLookupUnit::current_has_result() const {
    // Phase 10B.3: REGISTERED unit->arbiter edge — committed state.
    return result_buffer_.current().valid;
}

WritebackEntry TLookupUnit::consume_result() {
    // Phase 10B.3: pure read — mutates nothing.
    return result_buffer_.current();
}

} // namespace gpu_sim
