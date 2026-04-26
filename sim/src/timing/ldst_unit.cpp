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
            // Try to push to FIFO. The FIFO write is observable to the
            // coalescing unit's same-tick evaluate (COMBINATIONAL edge).
            if (next_addr_gen_fifo_.size() < fifo_depth_) {
                next_addr_gen_fifo_.push_back(next_pending_entry_);
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
    // next_addr_gen_fifo_ is the live FIFO read combinationally by
    // CoalescingUnit; nothing to copy.
}

void LdStUnit::reset() {
    current_busy_ = false;
    next_busy_ = false;
    current_cycles_remaining_ = 0;
    next_cycles_remaining_ = 0;
    current_pending_entry_.valid = false;
    next_pending_entry_.valid = false;
    next_addr_gen_fifo_.clear();
}

bool LdStUnit::has_result() const {
    // LD/ST unit doesn't produce results through writeback buffer directly.
    // Load results come from cache/MSHR fill path.
    return false;
}

WritebackEntry LdStUnit::consume_result() {
    return WritebackEntry{};  // Never called
}

} // namespace gpu_sim
