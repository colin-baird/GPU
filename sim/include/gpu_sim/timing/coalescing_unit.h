#pragma once

#include "gpu_sim/timing/ldst_unit.h"
#include "gpu_sim/timing/cache.h"
#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/stats.h"

namespace gpu_sim {

class CoalescingUnit {
public:
    CoalescingUnit(LdStUnit& ldst, L1Cache& cache, uint32_t line_size, Stats& stats);

    // Returns a writeback entry if a cache hit produced one
    void evaluate(WritebackEntry& wb_out, bool& wb_valid);
    void commit();
    void reset();

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
};

} // namespace gpu_sim
