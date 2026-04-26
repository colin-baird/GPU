#pragma once

#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/timing/operand_collector.h"
#include "gpu_sim/stats.h"
#include <deque>

namespace gpu_sim {

struct AddrGenFIFOEntry {
    bool valid = false;
    uint32_t warp_id;
    uint8_t dest_reg;
    bool is_load;
    bool is_store;
    TraceEvent trace;  // Contains addresses, store data, etc.
    uint64_t issue_cycle;
};

class LdStUnit : public ExecutionUnit {
public:
    LdStUnit(uint32_t num_ldst_units, uint32_t fifo_depth, Stats& stats);

    bool ready_out() const override { return !current_busy_; }
    void evaluate() override;
    void commit() override;
    void reset() override;
    bool has_result() const override;
    WritebackEntry consume_result() override;
    ExecUnit get_type() const override { return ExecUnit::LDST; }

    void accept(const DispatchInput& input, uint64_t cycle);

    // Coalescing/cache interface pulls entries from the FIFO same-tick after
    // LdStUnit::evaluate -- COMBINATIONAL edge. These accessors expose the
    // live (next_*) FIFO so a freshly-pushed entry is visible to coalescing
    // in the same tick (preserving zero cycle delta with pre-Phase-1 code).
    bool fifo_empty() const { return next_addr_gen_fifo_.empty(); }
    const AddrGenFIFOEntry& fifo_front() const { return next_addr_gen_fifo_.front(); }
    void fifo_pop() { next_addr_gen_fifo_.pop_front(); }
    bool busy() const { return current_busy_; }
    uint32_t cycles_remaining() const { return current_cycles_remaining_; }
    std::optional<uint32_t> active_warp() const {
        if (!current_busy_) return std::nullopt;
        return current_pending_entry_.warp_id;
    }
    const AddrGenFIFOEntry* pending_entry() const {
        return current_pending_entry_.valid ? &current_pending_entry_ : nullptr;
    }
    // Live FIFO view for coalescing-related trace dumps. Reads next_* so the
    // entry just pushed by this tick's evaluate is visible alongside the rest.
    const std::deque<AddrGenFIFOEntry>& fifo_entries() const { return next_addr_gen_fifo_; }

private:
    uint32_t num_ldst_units_;
    uint32_t fifo_depth_;
    Stats& stats_;

    // Phase 1 discipline: every cross-cycle field is double-buffered.
    bool current_busy_ = false;
    bool next_busy_ = false;
    uint32_t current_cycles_remaining_ = 0;
    uint32_t next_cycles_remaining_ = 0;
    AddrGenFIFOEntry current_pending_entry_;
    AddrGenFIFOEntry next_pending_entry_;
    // The address-generation FIFO is read by CoalescingUnit later in the same
    // tick (COMBINATIONAL edge), so accessors expose the live next_* deque and
    // there is no current_* mirror — commit() simply leaves next_addr_gen_fifo_
    // unchanged for the following tick.
    std::deque<AddrGenFIFOEntry> next_addr_gen_fifo_;
};

} // namespace gpu_sim
