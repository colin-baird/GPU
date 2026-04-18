#pragma once

#include "gpu_sim/timing/ldst_unit.h"
#include "gpu_sim/timing/cache.h"
#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/timing/load_gather_buffer.h"
#include "gpu_sim/stats.h"
#include <optional>

namespace gpu_sim {

class CoalescingUnit {
public:
    CoalescingUnit(LdStUnit& ldst, L1Cache& cache, LoadGatherBufferFile& gather_file,
                   uint32_t line_size, Stats& stats);

    // Pulls the next load/store FIFO entry and drives cache transactions.
    // Loads never produce writebacks here — the gather buffer emits them.
    void evaluate();
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
    LoadGatherBufferFile& gather_file_;
    uint32_t line_size_;
    Stats& stats_;

    // Current entry being processed
    bool processing_ = false;
    AddrGenFIFOEntry current_entry_;
    bool is_coalesced_ = false;
    uint32_t serial_index_ = 0;  // For serialized requests
};

} // namespace gpu_sim
