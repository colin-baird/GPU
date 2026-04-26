#pragma once

#include "gpu_sim/types.h"
#include "gpu_sim/timing/mshr.h"
#include "gpu_sim/timing/memory_interface.h"
#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/timing/load_gather_buffer.h"
#include "gpu_sim/stats.h"
#include <vector>
#include <deque>

namespace gpu_sim {

struct CacheTag {
    bool valid = false;
    uint32_t tag = 0;
    // Set when a primary fill installs a line and has a non-empty dependent
    // chain. A different-tag miss into a pinned set stalls with LINE_PINNED.
    bool pinned = false;
};

struct PendingCacheFill {
    bool valid = false;
    MemoryResponse response{};
};

enum class CacheStallReason {
    NONE,
    MSHR_FULL,
    WRITE_BUFFER_FULL,
    LINE_PINNED
};

struct CacheMissTraceEvent {
    bool valid = false;
    uint32_t warp_id = 0;
    uint32_t line_addr = 0;
    bool is_store = false;
    uint32_t pc = 0;
    uint32_t raw_instruction = 0;
    // True when this miss was allocated as a secondary MSHR (same-line merge).
    bool merged_secondary = false;
};

struct CacheFillTraceEvent {
    bool valid = false;
    uint32_t warp_id = 0;
    uint32_t line_addr = 0;
    bool is_store = false;
    uint32_t pc = 0;
    uint32_t raw_instruction = 0;
    // Length of the dependent chain (primary + secondaries) at fill time.
    uint32_t chain_length_at_fill = 0;
    // True when complete_fill deferred because the target set was pinned by a
    // different line. The fill stays in pending_fill_ and is retried next cycle.
    bool deferred = false;
};

struct CacheSecondaryDrainTraceEvent {
    bool valid = false;
    uint32_t warp_id = 0;
    uint32_t line_addr = 0;
    bool is_store = false;
    uint32_t pc = 0;
    uint32_t raw_instruction = 0;
};

struct CachePinStallTraceEvent {
    bool valid = false;
    uint32_t warp_id = 0;
    uint32_t requested_line_addr = 0;
    uint32_t pinned_line_addr = 0;
    bool is_store = false;
};

class L1Cache {
public:
    L1Cache(uint32_t cache_size, uint32_t line_size, uint32_t num_mshrs,
            uint32_t write_buffer_depth, ExternalMemoryInterface& mem_if,
            LoadGatherBufferFile& gather_file, Stats& stats);

    // Process one cache request. Returns true if the request was accepted.
    // For loads: on a hit, attempts to write the lanes selected by `lane_mask`
    // into the caller's gather buffer; returns false if the gather-buffer port
    // was already used this cycle (caller must retry next cycle). On a miss,
    // allocates an MSHR recording `lane_mask`. For stores: hit updates cache
    // and write buffer; miss allocates an MSHR (write-allocate).
    bool process_load(uint32_t addr, uint32_t warp_id, uint32_t lane_mask,
                      const std::array<uint32_t, WARP_SIZE>& results,
                      uint64_t issue_cycle, uint32_t pc, uint32_t raw_instruction);
    bool process_store(uint32_t line_addr, uint32_t warp_id, uint64_t issue_cycle,
                       uint32_t pc, uint32_t raw_instruction);

    // Handle one pending memory response (MSHR fill) per cycle. Load fills
    // deposit lane data into the owning warp's gather buffer; store fills push
    // the line into the write buffer.
    void handle_responses();

    // Drain write buffer
    void drain_write_buffer();

    void evaluate();
    void commit();
    void reset();

    bool is_stalled() const { return stalled_; }
    bool is_idle() const;
    CacheStallReason stall_reason() const { return stall_reason_; }
    uint32_t active_mshr_count() const;
    std::vector<uint32_t> active_mshr_warps() const;
    size_t write_buffer_size() const { return write_buffer_.size(); }
    const PendingCacheFill& pending_fill() const { return pending_fill_; }
    const MSHRFile& mshrs() const { return mshrs_; }
    const CacheMissTraceEvent& last_miss_event() const { return last_miss_event_; }
    const CacheFillTraceEvent& last_fill_event() const { return last_fill_event_; }
    const CacheSecondaryDrainTraceEvent& last_drain_event() const { return last_drain_event_; }
    const CachePinStallTraceEvent& last_pin_stall_event() const { return last_pin_stall_event_; }
    uint32_t pinned_line_count() const;
    uint32_t secondary_mshr_count() const;

private:
    bool complete_fill(const MemoryResponse& resp);
    void drain_secondary_chain_head();
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
    LoadGatherBufferFile& gather_file_;
    Stats& stats_;
    bool stalled_ = false;
    CacheStallReason stall_reason_ = CacheStallReason::NONE;
    PendingCacheFill pending_fill_;
    CacheMissTraceEvent last_miss_event_;
    CacheFillTraceEvent last_fill_event_;
    CacheSecondaryDrainTraceEvent last_drain_event_;
    CachePinStallTraceEvent last_pin_stall_event_;
    // Phase 7: the prior `gather_extract_port_used_` scratch flag has been
    // removed. The shared gather-extract port (spec §5.3 — one line-to-
    // gather-buffer extraction per cycle) is now arbitrated by
    // `LoadGatherBufferFile` via a single REGISTERED `next_port_claimed_` /
    // `current_port_claimed_` pair; the cache observes claim outcomes
    // through `try_write()`'s bool return. The FILL > secondary > HIT
    // priority is encoded in tick order: cache_->evaluate() runs at the top
    // of the non-panic tick (FILL first via handle_responses, secondary
    // second via drain_secondary_chain_head); coalescing_->evaluate() runs
    // later in the tick (HIT third).
};

} // namespace gpu_sim
