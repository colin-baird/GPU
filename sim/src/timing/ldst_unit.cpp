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
    accepted_this_cycle_.drive(true);
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
    busy_this_cycle_.drive(busy_.next());
    if (busy_.next()) {
        if (cycles_remaining_.next() > 0) {
            cycles_remaining_.next_mut()--;
        }
        if (cycles_remaining_.next() == 0) {
            // Phase 3 (close-the-Reg-family-migration): cross-stage addr-gen
            // FIFO. Stage the push on the TimingModel-owned RegFifo
            // conditioned on !writeback_stall — the literal simulator
            // translation of the RTL wr_en && !stall AND-gate. On a stalled
            // cycle stage_push is skipped, this stage's commit_all() also
            // early-returns, and the producer's pipeline registers
            // (busy_/cycles_remaining_/pending_entry_) hold; the next-cycle
            // evaluate() re-stages the push identically. The eligibility
            // check reads current_size() (the stable committed-cycle FIFO
            // depth) and ignores the consumer's same-cycle stage_pop, exactly
            // mirroring the pre-migration "ignore same-cycle pop_front()"
            // semantics — a one-cycle bubble when the FIFO is full and the
            // consumer pops in the same tick is documented parity.
            const bool wb_stall =
                wb_arbiter_ != nullptr && wb_arbiter_->next_writeback_stall();
            const std::size_t fifo_size =
                addr_gen_fifo_ != nullptr ? addr_gen_fifo_->current_size() : 0;
            if (!wb_stall && fifo_size < fifo_depth_) {
                if (addr_gen_fifo_ != nullptr) {
                    addr_gen_fifo_->stage_push(pending_entry_.next());
                }
                push_staged_this_cycle_.drive(true);
                pending_entry_.next_mut().valid = false;
                busy_.set_next(false);
            }
            // If FIFO full or writeback-stalled, stay busy (stall) until a
            // slot opens or the stall releases.
        }
    }
}

void LdStUnit::commit() {
    // Phase 10B.3: writeback-stall self-gate. LdStUnit is one of the five
    // issue/execute units that freeze on a writeback-stall cycle: no
    // commit_all() flip, no fifo_total_pushes_ increment, so the next tick's
    // seed_next()+evaluate() re-stages the push identically. Coalescing
    // (ungated) may still pop from the cross-stage FIFO this cycle via the
    // TimingModel-owned commit pass, which simply shrinks it; the held push
    // lands on the resumed cycle. Note that push_staged_this_cycle_ is also
    // gated at evaluate() on the same !writeback_stall, so on a stall this
    // wire is naturally de-asserted (the producer never staged the push).
    if (wb_arbiter_ != nullptr && wb_arbiter_->next_writeback_stall()) {
        return;
    }

    // Phase 10B.0.5: Stats increments relocated here from evaluate()/accept().
    // Both per-cycle flags are consumed and cleared at commit() so a commit()
    // not preceded by an evaluate() never re-counts a stale flag.
    if (busy_this_cycle_.value()) {
        stats_.ldst_stats.busy_cycles++;
    }
    if (accepted_this_cycle_.value()) {
        stats_.ldst_stats.instructions++;
    }
    busy_this_cycle_.reset();
    accepted_this_cycle_.reset();

    // Phase 3 (close-the-Reg-family-migration): the push lands at the cross-
    // stage FIFO commit pass (TimingModel-owned, ungated, sequenced after
    // this commit). Advance the monotonic push counter in lockstep with the
    // push intent — the gating ensures push_staged_this_cycle_ was driven iff
    // the cross-stage commit pass will apply the push to the queue. The
    // scheduler's LDST FIFO-occupancy gate reads current_fifo_total_pushes()
    // on the next tick (back-to-front sweep: scheduler.evaluate is sequenced
    // after this commit), so the visibility timing matches the pre-migration
    // commit-time-increment semantics byte-identically.
    if (push_staged_this_cycle_.value()) {
        ++fifo_total_pushes_;
    }
    push_staged_this_cycle_.reset();

    commit_all();
}

void LdStUnit::reset() {
    reset_all();
    // The cross-stage addr-gen FIFO is owned by TimingModel and reset by
    // TimingModel; do not clear it here.
    // Phase 10B.0: cleared in lockstep with WarpScheduler::reset()'s
    // ldst_issued_total_ (both run in the panic-flush cascade) so the
    // (issued - pushed) difference restarts at zero.
    fifo_total_pushes_ = 0;
    busy_this_cycle_.reset();
    accepted_this_cycle_.reset();
    push_staged_this_cycle_.reset();
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
