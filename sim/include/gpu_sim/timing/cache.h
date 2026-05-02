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

// Phase M3: REGISTERED forward command path between coalescing and cache.
struct LoadCommand {
    bool valid = false;
    uint32_t addr = 0;
    uint32_t warp_id = 0;
    uint32_t lane_mask = 0;
    std::array<uint32_t, WARP_SIZE> results{};
    uint64_t issue_cycle = 0;
    uint32_t pc = 0;
    uint32_t raw_instruction = 0;
};

struct StoreCommand {
    bool valid = false;
    uint32_t line_addr = 0;
    uint32_t warp_id = 0;
    uint64_t issue_cycle = 0;
    uint32_t pc = 0;
    uint32_t raw_instruction = 0;
};

class L1Cache {
public:
    L1Cache(uint32_t cache_size, uint32_t line_size, uint32_t num_mshrs,
            uint32_t write_buffer_depth, ExternalMemoryInterface& mem_if,
            LoadGatherBufferFile& gather_file, Stats& stats);

    // Direct synchronous API. Used by tests and (until Phase M3 lands the
    // production rewrite) called internally by L1Cache::evaluate to process
    // the REGISTERED current_load_cmd_ / current_store_cmd_. Returns true
    // when the request was accepted; tests rely on this bool. Production
    // coalescing now goes through set_next_load_cmd / set_next_store_cmd
    // gated by next_cmd_stall().
    bool process_load(uint32_t addr, uint32_t warp_id, uint32_t lane_mask,
                      const std::array<uint32_t, WARP_SIZE>& results,
                      uint64_t issue_cycle, uint32_t pc, uint32_t raw_instruction);
    bool process_store(uint32_t line_addr, uint32_t warp_id, uint64_t issue_cycle,
                       uint32_t pc, uint32_t raw_instruction);

    // Phase M3: REGISTERED command setters. Coalescing writes next_*_cmd_;
    // commit() flips into current_*_cmd_; evaluate() processes the cmd and
    // clears the valid bit. The bool return is gone — the cmd is guaranteed
    // acceptable when received because next_cmd_stall() at submit time was
    // false.
    void set_next_load_cmd(uint32_t addr, uint32_t warp_id, uint32_t lane_mask,
                           const std::array<uint32_t, WARP_SIZE>& results,
                           uint64_t issue_cycle, uint32_t pc, uint32_t raw_instruction);
    void set_next_store_cmd(uint32_t line_addr, uint32_t warp_id, uint64_t issue_cycle,
                            uint32_t pc, uint32_t raw_instruction);

    // Phase M3: COMBINATIONAL backward stall signal. Coalescing reads this
    // same-cycle (cache.evaluate runs before coalescing.evaluate in tick
    // order) before deciding whether to submit a cmd. Conservative: returns
    // true when ANY of the cmd-acceptance failure modes might fire at next
    // cycle's cmd processing. Lost throughput is bounded; correctness is
    // guaranteed.
    bool next_cmd_stall() const;
    // Companion accessor: which cmd-stall condition fires this cycle (for
    // trace classification). Returns NONE when next_cmd_stall() is false.
    // Order: MSHR_FULL > WRITE_BUFFER_FULL > LINE_PINNED > NONE
    // (FILL/secondary port conflicts surface as NONE since they are not
    // a structural stall the trace needs to report — the warp is just
    // waiting on memory).
    CacheStallReason next_cmd_stall_reason() const;

    // Handle one pending memory response (MSHR fill) per cycle. Load fills
    // deposit lane data into the owning warp's gather buffer; store fills push
    // the line into the write buffer.
    void handle_responses();

    // Drain write buffer
    void drain_write_buffer();

    void evaluate();
    void commit();
    void reset();

    // Phase 9 cache external boundary: each accessor declares its
    // discipline class. See resources/timing_discipline.md.
    //
    // COMBINATIONAL same-tick (single-slot, direct mutation, reset at top
    // of evaluate). Coalescing reads next_stalled() / next_stall_reason()
    // during its own evaluate, after cache.evaluate has already produced
    // this cycle's stall outcome. Models a same-cycle backpressure handshake
    // where the cache's stall signal is combinationally driven from
    // registered tag / write-buffer / pending_fill state.
    bool next_stalled() const { return stalled_; }
    CacheStallReason next_stall_reason() const { return stall_reason_; }
    //
    // REGISTERED (next/current pair flipped by commit()). Trace recording
    // runs at end-of-tick after cache.commit(); reads return committed
    // state. pending_fill_ also persists across cycles as a deferred-fill
    // carrier.
    bool is_idle() const;
    uint32_t active_mshr_count() const;
    std::vector<uint32_t> active_mshr_warps() const;
    size_t write_buffer_size() const { return write_buffer_.size(); }
    const PendingCacheFill& current_pending_fill() const { return current_pending_fill_; }
    const MSHRFile& mshrs() const { return mshrs_; }
    const CacheMissTraceEvent& current_last_miss_event() const { return current_last_miss_event_; }
    const CacheFillTraceEvent& current_last_fill_event() const { return current_last_fill_event_; }
    const CacheSecondaryDrainTraceEvent& current_last_drain_event() const { return current_last_drain_event_; }
    const CachePinStallTraceEvent& current_last_pin_stall_event() const { return current_last_pin_stall_event_; }
    uint32_t pinned_line_count() const;
    uint32_t secondary_mshr_count() const;

private:
    bool complete_fill(const MemoryResponse& resp);
    void drain_secondary_chain_head();
    uint32_t get_set(uint32_t addr) const;
    uint32_t get_tag(uint32_t addr) const;
    uint32_t get_line_addr(uint32_t addr) const;
    bool any_pinned_tag() const;

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
    // COMBINATIONAL same-tick scratch: reset at top of evaluate(), written
    // by handle_responses / process_load / process_store / complete_fill,
    // observed mid-tick by CoalescingUnit::evaluate. No next/current pair —
    // coalescing reads this cycle's value to make a same-cycle bail
    // decision (single-cycle backpressure path; tags_/write_buffer_/
    // pending_fill_ all source from registered state).
    bool stalled_ = false;
    CacheStallReason stall_reason_ = CacheStallReason::NONE;
    // REGISTERED next/current pairs (flipped by commit()).
    PendingCacheFill current_pending_fill_;
    PendingCacheFill next_pending_fill_;
    CacheMissTraceEvent current_last_miss_event_;
    CacheMissTraceEvent next_last_miss_event_;
    CacheFillTraceEvent current_last_fill_event_;
    CacheFillTraceEvent next_last_fill_event_;
    CacheSecondaryDrainTraceEvent current_last_drain_event_;
    CacheSecondaryDrainTraceEvent next_last_drain_event_;
    CachePinStallTraceEvent current_last_pin_stall_event_;
    CachePinStallTraceEvent next_last_pin_stall_event_;
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

    // Phase M3: REGISTERED forward command slots. Coalescing writes
    // next_load_cmd_ / next_store_cmd_; commit() flips next → current;
    // evaluate() consumes current_load_cmd_ / current_store_cmd_ after
    // handle_responses + drain_secondary_chain_head (HIT slot in the
    // FILL > secondary > HIT priority ladder).
    LoadCommand current_load_cmd_;
    LoadCommand next_load_cmd_;
    StoreCommand current_store_cmd_;
    StoreCommand next_store_cmd_;
};

} // namespace gpu_sim
