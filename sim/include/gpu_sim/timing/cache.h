#pragma once

#include "gpu_sim/types.h"
#include "gpu_sim/timing/mshr.h"
#include "gpu_sim/timing/memory_interface.h"
#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/stats.h"
#include <vector>
#include <deque>

namespace gpu_sim {

struct CacheTag {
    bool valid = false;
    uint32_t tag = 0;
};

class L1Cache {
public:
    L1Cache(uint32_t cache_size, uint32_t line_size, uint32_t num_mshrs,
            uint32_t write_buffer_depth, ExternalMemoryInterface& mem_if, Stats& stats);

    // Process one cache request. Returns true if the request was accepted.
    // For loads: if hit, fills wb_out. If miss, allocates MSHR.
    // For stores: if hit, updates cache and write buffer. If miss, allocates MSHR.
    bool process_load(uint32_t addr, uint32_t warp_id, uint8_t dest_reg,
                      const std::array<uint32_t, WARP_SIZE>& results,
                      uint64_t issue_cycle, WritebackEntry& wb_out);
    bool process_store(uint32_t line_addr);

    // Handle memory responses (MSHR fills)
    void handle_responses(WritebackEntry& wb_out, bool& wb_valid);

    // Drain write buffer
    void drain_write_buffer();

    void evaluate();
    void commit();
    void reset();

    bool is_stalled() const { return stalled_; }

private:
    uint32_t get_set(uint32_t addr) const;
    uint32_t get_tag(uint32_t addr) const;
    uint32_t get_line_addr(uint32_t addr) const;

    uint32_t cache_size_;
    uint32_t line_size_;
    uint32_t num_sets_;

    std::vector<CacheTag> tags_;
    MSHRFile mshrs_;
    std::deque<uint32_t> write_buffer_;  // Queue of line addresses to write back
    uint32_t write_buffer_depth_;
    ExternalMemoryInterface& mem_if_;
    Stats& stats_;
    bool stalled_ = false;
};

} // namespace gpu_sim
