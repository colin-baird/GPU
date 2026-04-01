#include "gpu_sim/timing/ldst_unit.h"

namespace gpu_sim {

LdStUnit::LdStUnit(uint32_t num_ldst_units, uint32_t fifo_depth, Stats& stats)
    : num_ldst_units_(num_ldst_units), fifo_depth_(fifo_depth), stats_(stats) {}

void LdStUnit::accept(const DispatchInput& input, uint64_t cycle) {
    busy_ = true;
    // ceil(32 / num_ldst_units) cycles for address generation
    cycles_remaining_ = (WARP_SIZE + num_ldst_units_ - 1) / num_ldst_units_;

    pending_entry_.valid = true;
    pending_entry_.warp_id = input.warp_id;
    pending_entry_.dest_reg = input.decoded.rd;
    pending_entry_.is_load = input.trace.is_load;
    pending_entry_.is_store = input.trace.is_store;
    pending_entry_.trace = input.trace;
    pending_entry_.issue_cycle = cycle;
    stats_.ldst_stats.instructions++;
}

void LdStUnit::evaluate() {
    if (busy_) {
        stats_.ldst_stats.busy_cycles++;
        if (cycles_remaining_ > 0) {
            cycles_remaining_--;
        }
        if (cycles_remaining_ == 0) {
            // Try to push to FIFO
            if (addr_gen_fifo_.size() < fifo_depth_) {
                addr_gen_fifo_.push_back(pending_entry_);
                pending_entry_.valid = false;
                busy_ = false;
            }
            // If FIFO full, stay busy (stall) until a slot opens
        }
    }
}

void LdStUnit::commit() {}

void LdStUnit::reset() {
    busy_ = false;
    cycles_remaining_ = 0;
    pending_entry_.valid = false;
    addr_gen_fifo_.clear();
}

bool LdStUnit::is_ready() const {
    return !busy_;
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
