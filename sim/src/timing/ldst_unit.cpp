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
    stats_.ldst_stats.instructions++;
}

void LdStUnit::evaluate() {
    // Operates on next_* (seeded equal to current_* by commit() at the end
    // of the prior tick, possibly updated by accept() earlier this tick).
    if (next_busy_) {
        stats_.ldst_stats.busy_cycles++;
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
    current_busy_ = next_busy_;
    current_cycles_remaining_ = next_cycles_remaining_;
    current_pending_entry_ = next_pending_entry_;
    // Phase M1: apply the staged push. Coalescing's commit (which runs
    // after this in TimingModel::tick()) calls pop_front() — push touches
    // the back, pop touches the front, so order between the two commits
    // is irrelevant for correctness.
    if (next_push_) {
        addr_gen_fifo_.push_back(*next_push_);
        next_push_.reset();
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
