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

    void evaluate() override;
    void commit() override;
    void reset() override;
    bool is_ready() const override;
    bool has_result() const override;
    WritebackEntry consume_result() override;
    ExecUnit get_type() const override { return ExecUnit::LDST; }

    void accept(const DispatchInput& input, uint64_t cycle);

    // Coalescing/cache interface pulls entries from the FIFO
    bool fifo_empty() const { return addr_gen_fifo_.empty(); }
    const AddrGenFIFOEntry& fifo_front() const { return addr_gen_fifo_.front(); }
    void fifo_pop() { addr_gen_fifo_.pop_front(); }
    bool busy() const { return busy_; }
    uint32_t cycles_remaining() const { return cycles_remaining_; }
    std::optional<uint32_t> active_warp() const {
        if (!busy_) return std::nullopt;
        return pending_entry_.warp_id;
    }
    const AddrGenFIFOEntry* pending_entry() const {
        return pending_entry_.valid ? &pending_entry_ : nullptr;
    }
    const std::deque<AddrGenFIFOEntry>& fifo_entries() const { return addr_gen_fifo_; }

private:
    uint32_t num_ldst_units_;
    uint32_t fifo_depth_;
    Stats& stats_;

    bool busy_ = false;
    uint32_t cycles_remaining_ = 0;
    AddrGenFIFOEntry pending_entry_;
    std::deque<AddrGenFIFOEntry> addr_gen_fifo_;
};

} // namespace gpu_sim
