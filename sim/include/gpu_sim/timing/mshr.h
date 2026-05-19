#pragma once

#include "gpu_sim/types.h"
#include "gpu_sim/trace_event.h"
#include "gpu_sim/timing/reg.h"
#include <array>
#include <cstdint>
#include <vector>

namespace gpu_sim {

struct MSHREntry {
    static constexpr uint32_t INVALID_MSHR = 0xFFFFFFFFu;

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
    // Same-line merging: singly-linked dependent chain in program order.
    // `next_in_chain` points to the next MSHR for the same line, or
    // INVALID_MSHR at the tail. `is_secondary` is true for entries that
    // inherit the primary's external fetch (no own submit_read).
    uint32_t next_in_chain = INVALID_MSHR;
    bool is_secondary = false;
};

// Registered (double-buffered) MSHR file. Readers scan the committed
// entries (`entries_.current()`); writers (allocate / free / chain-link)
// mutate the staged entries (`entries_.next_mut()`); `commit()` flips.
// A slot freed this cycle is therefore not reusable until the next cycle,
// and a chain tail freed this cycle is still visible to a same-cycle
// `find_chain_tail`. The L1 cache drives `seed_next()` / `commit()` from
// its own evaluate-top / commit hooks.
//
// Phase 5b (reg.h migration): MSHRFile is an internal helper buffer (lint's
// INTERNAL_HELPER_CLASSES) that derives RegisteredStage for the mixin's
// seed_all() / commit_all() / reset_all() loops only — it is NOT a pipeline
// stage and is not enrolled in TimingModel::tick()'s seed phase. The entire
// `std::vector<MSHREntry>` is wrapped as one Reg (whole-vector wrapping,
// identical to the cache's Reg<std::vector<CacheTag>>), so seed/commit
// compile to the same vector assignment the hand-rolled double-buffer did.
class MSHRFile : public RegisteredStage {
public:
    explicit MSHRFile(uint32_t num_entries);

    // Returns MSHR index, or -1 if no free entry. Scans the committed
    // entries for the free slot, writes the entry into the staged entries
    // (registered).
    int allocate(const MSHREntry& entry);

    // Free an MSHR entry — clears the valid bit in the staged entries.
    void free(uint32_t index);

    // Access by index. current_at: committed read (REGISTERED). next_at:
    // mutable write into the next-state slot (e.g. a chain-link write).
    const MSHREntry& current_at(uint32_t index) const { return entries_.current()[index]; }
    MSHREntry& next_at(uint32_t index) { return entries_.next_mut()[index]; }

    bool has_free() const;
    bool has_active() const;

    // Linear scan over valid committed entries matching `line_addr`. Returns
    // the MSHR index of the chain tail (the entry whose `next_in_chain ==
    // INVALID_MSHR`) for this line, or -1 if no MSHR currently holds it.
    int find_chain_tail(uint32_t line_addr) const;
    uint32_t num_entries() const { return num_entries_; }

    // Double-buffer lifecycle, driven by L1Cache. Phase 5b: delegates to
    // RegisteredStage's seed_all() / commit_all() / reset_all().
    void seed_next();   // entries_.next = entries_.current (top of tick)
    void commit();      // entries_.current = entries_.next (cycle boundary)
    void reset();

private:
    uint32_t num_entries_;  // config
    // REGISTERED MSHR file. The whole vector is one register: a Reg's seed()
    // / commit() compile to the same std::vector assignment as the
    // pre-Phase-5b hand-rolled `next_entries_ = current_entries_` /
    // `current_entries_ = next_entries_`.
    Reg<std::vector<MSHREntry>> entries_;
};

} // namespace gpu_sim
