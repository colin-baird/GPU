#include "gpu_sim/timing/ldst_unit.h"
#include "gpu_sim/timing/writeback_arbiter.h"

namespace gpu_sim {

LdStUnit::LdStUnit(uint32_t num_ldst_units, uint32_t fifo_depth, Stats& stats)
    : num_ldst_units_(num_ldst_units), fifo_depth_(fifo_depth), stats_(stats) {
    register_state(&busy_, &cycles_remaining_, &pending_entry_);
}

void LdStUnit::accept(const DispatchInput& input, uint64_t cycle) {
    // Phase 1 discipline: writes only into the staged slot. ceil(32 / num_ldst_units)
    // cycles for address generation.
    busy_.set_next(true);
    cycles_remaining_.set_next((WARP_SIZE + num_ldst_units_ - 1) / num_ldst_units_);

    AddrGenFIFOEntry& pe = pending_entry_.next_mut();
    pe.valid = true;
    pe.warp_id = input.warp_id;
    pe.dest_reg = input.decoded.rd;
    pe.is_load = input.trace.is_load;
    pe.is_store = input.trace.is_store;
    pe.trace = input.trace;
    pe.issue_cycle = cycle;
    // Phase 10B.0.5: ldst_stats.instructions is incremented in commit()
    // (gated on accepted_this_cycle_), not here.
    accepted_this_cycle_ = true;
}

void LdStUnit::evaluate() {
    // Phase 10B.1: REGISTERED opcoll->unit edge via the pull model. Read
    // opcoll's committed output and self-select by target_unit.
    if (opcoll_ != nullptr) {
        const auto& dispatched = opcoll_->current_output();
        if (dispatched && dispatched->decoded.target_unit == ExecUnit::LDST) {
            accept(*dispatched, sim_cycle_ != nullptr ? *sim_cycle_ : 0);
        }
    }

    // Operates on the staged slots (seeded equal to current_* by seed_next()
    // at the top of the tick, possibly updated by accept() earlier this tick).
    // Phase 10B.0.5: capture the per-cycle busy flag before the body may
    // clear next_busy_; ldst_stats.busy_cycles is incremented at commit().
    busy_this_cycle_ = busy_.next();
    if (busy_.next()) {
        if (cycles_remaining_.next() > 0) {
            cycles_remaining_.next_mut()--;
        }
        if (cycles_remaining_.next() == 0) {
            // Phase M1: REGISTERED FIFO. Stage the push in next_push_; the
            // commit phase applies it (gated by the writeback stall). The
            // eligibility check uses the stable committed-cycle FIFO size;
            // this mirrors fetch_stage.cpp's will_be_full check that does not
            // account for the consumer's same-cycle pop. A one-cycle bubble
            // when the FIFO is full and coalescing pops same-tick is the
            // documented parity.
            if (addr_gen_fifo_.size() < fifo_depth_) {
                next_push_ = pending_entry_.next();
                pending_entry_.next_mut().valid = false;
                busy_.set_next(false);
            }
            // If FIFO full, stay busy (stall) until a slot opens
        }
    }
}

void LdStUnit::commit() {
    // Phase 10B.3: writeback-stall self-gate. LdStUnit is one of the five
    // issue/execute units that freeze on a writeback-stall cycle: no
    // commit_all() flip and no next_push_ application, so the next tick's
    // seed_next()+evaluate() re-stages the push identically. Coalescing
    // (ungated) may still pop from the FIFO this cycle, which simply
    // shrinks it; the held push lands on the resumed cycle.
    if (wb_arbiter_ != nullptr && wb_arbiter_->next_writeback_stall()) {
        return;
    }

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

    // Phase M1: apply the staged push to the committed FIFO; advance the
    // monotonic push counter in lockstep with the push (the scheduler's
    // LDST FIFO-occupancy gate reads current_fifo_total_pushes()).
    if (next_push_) {
        addr_gen_fifo_.push_back(*next_push_);
        next_push_.reset();
        ++fifo_total_pushes_;
    }

    commit_all();
}

void LdStUnit::reset() {
    reset_all();
    addr_gen_fifo_.clear();
    next_push_.reset();
    // Phase 10B.0: cleared in lockstep with WarpScheduler::reset()'s
    // ldst_issued_total_ (both run in the panic-flush cascade) so the
    // (issued - pushed) difference restarts at zero.
    fifo_total_pushes_ = 0;
    busy_this_cycle_ = false;
    accepted_this_cycle_ = false;
}

bool LdStUnit::current_has_result() const {
    // LD/ST unit doesn't produce results through writeback buffer directly.
    // Load results come from cache/MSHR fill path (the LoadGatherBufferFile,
    // which is the separate writeback arbiter source). LdStUnit is not
    // registered with the arbiter.
    return false;
}

WritebackEntry LdStUnit::consume_result() {
    return WritebackEntry{};  // Never called
}

} // namespace gpu_sim
