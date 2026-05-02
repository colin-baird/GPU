#pragma once

#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/timing/operand_collector.h"
#include "gpu_sim/stats.h"
#include <deque>
#include <optional>

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

    bool current_busy() const override { return current_busy_; }
    void evaluate() override;
    void commit() override;
    void reset() override;
    bool next_has_result() const override;
    WritebackEntry consume_result() override;
    ExecUnit get_type() const override { return ExecUnit::LDST; }

    void accept(const DispatchInput& input, uint64_t cycle);

    // Phase M1 discipline: the address-gen FIFO is REGISTERED. evaluate()
    // does not push directly; it stages an entry in next_push_ and commit()
    // applies it. coalescing reads current_fifo_* (the stable single-field
    // state) during its evaluate and stages a pop in its own next-side flag,
    // applied at coalescing.commit() via pop_front(). End-of-cycle state is
    // identical to the pre-M1 mid-evaluate model. This matches the structural
    // pattern in fetch_stage.cpp where fetch's `will_be_full` check does not
    // account for scheduler's same-cycle pop of `instr_buffer`; an FIFO-full
    // bubble of one cycle is parity with that pattern.
    bool current_fifo_empty() const { return addr_gen_fifo_.empty(); }
    const AddrGenFIFOEntry& current_fifo_front() const { return addr_gen_fifo_.front(); }
    void pop_front() { addr_gen_fifo_.pop_front(); }
    bool busy() const { return current_busy_; }
    uint32_t current_cycles_remaining() const { return current_cycles_remaining_; }
    std::optional<uint32_t> active_warp() const {
        if (!current_busy_) return std::nullopt;
        return current_pending_entry_.warp_id;
    }
    const AddrGenFIFOEntry* pending_entry() const {
        return current_pending_entry_.valid ? &current_pending_entry_ : nullptr;
    }
    // FIFO view for trace dumps. Reads the stable REGISTERED deque (same
    // state observed by coalescing during evaluate this cycle).
    const std::deque<AddrGenFIFOEntry>& current_fifo_entries() const { return addr_gen_fifo_; }
    // Size accessor for upstream issue-gate bookkeeping (consumed by the
    // pipeline plan's Phase 10B.0 LDST FIFO accounting).
    uint32_t current_fifo_size() const { return static_cast<uint32_t>(addr_gen_fifo_.size()); }

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
    // Phase M1: REGISTERED address-gen FIFO. Mutated only at commit phase
    // — producer (this unit) writes next_push_ at evaluate; consumer
    // (coalescing) drives pop_front() from its own commit. Reads during
    // evaluate see the start-of-cycle state.
    std::deque<AddrGenFIFOEntry> addr_gen_fifo_;
    std::optional<AddrGenFIFOEntry> next_push_;
};

} // namespace gpu_sim
