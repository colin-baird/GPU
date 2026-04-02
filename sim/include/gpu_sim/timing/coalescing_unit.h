#pragma once

#include "gpu_sim/timing/ldst_unit.h"
#include "gpu_sim/timing/cache.h"
#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/stats.h"
#include <optional>

namespace gpu_sim {

class CoalescingUnit {
public:
    CoalescingUnit(LdStUnit& ldst, L1Cache& cache, uint32_t line_size, Stats& stats);

    // Returns a writeback entry if a cache hit produced one
    void evaluate(WritebackEntry& wb_out, bool& wb_valid);
    void commit();
    void reset();
    bool is_idle() const { return !processing_; }
    std::optional<uint32_t> active_warp() const {
        if (!processing_) return std::nullopt;
        return current_entry_.warp_id;
    }
    bool is_coalesced() const { return is_coalesced_; }
    uint32_t serial_index() const { return serial_index_; }
    const AddrGenFIFOEntry* current_entry() const {
        return processing_ ? &current_entry_ : nullptr;
    }

private:
    LdStUnit& ldst_;
    L1Cache& cache_;
    uint32_t line_size_;
    Stats& stats_;

    // Current entry being processed
    bool processing_ = false;
    AddrGenFIFOEntry current_entry_;
    bool is_coalesced_ = false;
    uint32_t serial_index_ = 0;  // For serialized requests
    bool wb_already_produced_ = false;  // Serialized load writeback suppression
};

} // namespace gpu_sim
