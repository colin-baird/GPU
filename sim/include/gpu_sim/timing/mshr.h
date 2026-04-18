#pragma once

#include "gpu_sim/types.h"
#include "gpu_sim/trace_event.h"
#include <array>
#include <cstdint>
#include <vector>

namespace gpu_sim {

struct MSHREntry {
    bool valid = false;
    uint32_t cache_line_addr = 0;
    bool is_store = false;
    uint32_t warp_id = 0;
    uint8_t dest_reg = 0;
    uint32_t pc = 0;
    uint32_t raw_instruction = 0;
    uint64_t issue_cycle = 0;
    // Per-thread data from the trace event
    std::array<uint32_t, WARP_SIZE> mem_addresses{};
    std::array<uint32_t, WARP_SIZE> store_data{};
    std::array<uint8_t, WARP_SIZE> mem_size{};
    // Loaded data result (filled by functional model already in trace)
    std::array<uint32_t, WARP_SIZE> results{};
    // For load misses: lanes of the owning warp waiting on this fill. The
    // load's destination register lives in the warp's gather buffer; on fill
    // the cache deposits these lanes into the gather buffer.
    uint32_t lane_mask = 0;
};

class MSHRFile {
public:
    explicit MSHRFile(uint32_t num_entries);

    // Returns MSHR index, or -1 if no free entry
    int allocate(const MSHREntry& entry);

    // Free an MSHR entry
    void free(uint32_t index);

    // Access by index
    MSHREntry& at(uint32_t index) { return entries_[index]; }
    const MSHREntry& at(uint32_t index) const { return entries_[index]; }

    bool has_free() const;
    bool has_active() const;
    uint32_t num_entries() const { return num_entries_; }

    void reset();

private:
    uint32_t num_entries_;
    std::vector<MSHREntry> entries_;
};

} // namespace gpu_sim
