#include "gpu_sim/timing/ldst_unit.h"

namespace gpu_sim {

LdStUnit::LdStUnit(uint32_t num_ldst_units, uint32_t fifo_depth, Stats& stats)
    : num_ldst_units_(num_ldst_units), fifo_depth_(fifo_depth), stats_(stats) {}

void LdStUnit::accept(const DispatchInput& input, uint64_t cycle) {
    // Phase 1 discipline: writes only into next_* slots. ceil(32 / num_ldst_units)
    // cycles for address generation.
    next_busy_ = true;
    next_cycles_remaining_ = (WARP_SIZE + num_ldst_units_ - 1) / num_ldst_units_;

    next_pending_entry_.valid = true;
    next_pending_entry_.warp_id = input.warp_id;
    next_pending_entry_.dest_reg = input.decoded.rd;
    next_pending_entry_.is_load = input.trace.is_load;
    next_pending_entry_.is_store = input.trace.is_store;
    next_pending_entry_.trace = input.trace;
    next_pending_entry_.issue_cycle = cycle;
    // Phase 10B.0.5: ldst_stats.instructions is incremented in commit()
    // (gated on accepted_this_cycle_), not here.
    accepted_this_cycle_ = true;
}

void LdStUnit::seed_next() {
    // Phase 10B.0.5: re-establish the carry-forward addr-gen state in next_*
    // at the top of the tick. addr_gen_fifo_ / fifo_total_pushes_ are not
    // double-buffered (M1 commit-phase-mutation discipline) and are untouched.
    next_busy_ = current_busy_;
    next_cycles_remaining_ = current_cycles_remaining_;
    next_pending_entry_ = current_pending_entry_;
}

void LdStUnit::evaluate() {
    // Operates on next_* (seeded equal to current_* by seed_next() at the top
    // of the tick, possibly updated by accept() earlier this tick).
    // Phase 10B.0.5: next_push_ is a fresh per-cycle staging slot — reset it
    // at the top of evaluate() (moved here from commit()) so a gated commit()
    // does not skip the reset. Matches how opcoll::evaluate resets
    // next_output_.
    next_push_.reset();
    // Phase 10B.0.5: capture the per-cycle busy flag before the body may
    // clear next_busy_; ldst_stats.busy_cycles is incremented at commit().
    busy_this_cycle_ = next_busy_;
    if (next_busy_) {
        if (next_cycles_remaining_ > 0) {
            next_cycles_remaining_--;
        }
        if (next_cycles_remaining_ == 0) {
            // Phase M1: REGISTERED FIFO. The push is staged in next_push_
            // and applied at commit(). Eligibility check uses the stable
            // current-cycle FIFO size; this mirrors fetch_stage.cpp's
            // will_be_full check that does not account for the consumer's
            // same-cycle pop. A one-cycle bubble when the FIFO is full and
            // coalescing pops same-tick is the documented parity.
            if (addr_gen_fifo_.size() < fifo_depth_) {
                next_push_ = next_pending_entry_;
                next_pending_entry_.valid = false;
                next_busy_ = false;
            }
            // If FIFO full, stay busy (stall) until a slot opens
        }
    }
}

void LdStUnit::commit() {
    // Phase 10B.0.5: Stats increments relocated here from evaluate()/accept().
    // Both per-cycle flags are consumed and cleared at commit() so a commit()
    // not preceded by an evaluate() never re-counts a stale flag.
    if (busy_this_cycle_) {
        stats_.ldst_stats.busy_cycles++;
        busy_this_cycle_ = false;
    }
    if (accepted_this_cycle_) {
        stats_.ldst_stats.instructions++;
        accepted_this_cycle_ = false;
    }

    current_busy_ = next_busy_;
    current_cycles_remaining_ = next_cycles_remaining_;
    current_pending_entry_ = next_pending_entry_;
    // Phase M1: apply the staged push. Coalescing's commit (which runs
    // after this in TimingModel::tick()) calls pop_front() — push touches
    // the back, pop touches the front, so order between the two commits
    // is irrelevant for correctness.
    // Phase 10B.0.5: next_push_ is no longer reset here — it is reset at the
    // top of evaluate() as a fresh per-cycle staging slot, so a gated commit()
    // cannot skip the reset.
    if (next_push_) {
        addr_gen_fifo_.push_back(*next_push_);
        // Phase 10B.0: monotonic push counter advances on the same cycle the
        // op becomes visible in addr_gen_fifo_, so the scheduler's
        // (issued - pushed) difference stays invariant across the FIFO-entry
        // transition.
        ++fifo_total_pushes_;
    }
}

void LdStUnit::reset() {
    current_busy_ = false;
    next_busy_ = false;
    current_cycles_remaining_ = 0;
    next_cycles_remaining_ = 0;
    current_pending_entry_.valid = false;
    next_pending_entry_.valid = false;
    addr_gen_fifo_.clear();
    next_push_.reset();
    // Phase 10B.0: cleared in lockstep with WarpScheduler::reset()'s
    // ldst_issued_total_ (both run in the panic-flush cascade) so the
    // (issued - pushed) difference restarts at zero.
    fifo_total_pushes_ = 0;
    busy_this_cycle_ = false;
    accepted_this_cycle_ = false;
}

bool LdStUnit::next_has_result() const {
    // LD/ST unit doesn't produce results through writeback buffer directly.
    // Load results come from cache/MSHR fill path.
    return false;
}

WritebackEntry LdStUnit::consume_result() {
    return WritebackEntry{};  // Never called
}

} // namespace gpu_sim
