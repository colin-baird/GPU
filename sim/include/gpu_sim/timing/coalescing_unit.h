#pragma once

#include "gpu_sim/timing/ldst_unit.h"
#include "gpu_sim/timing/cache.h"
#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/timing/load_gather_buffer.h"
#include "gpu_sim/stats.h"
#include <optional>

namespace gpu_sim {

class CoalescingUnit : public RegisteredStage {
public:
    CoalescingUnit(LdStUnit& ldst, L1Cache& cache, LoadGatherBufferFile& gather_file,
                   uint32_t line_size, Stats& stats);

    // Pulls the next load/store FIFO entry and drives cache transactions.
    // Loads never produce writebacks here — the gather buffer emits them.
    void seed_next() { seed_all(); }
    void evaluate();
    void commit();
    void reset();
    bool is_idle() const { return !processing_.current(); }
    std::optional<uint32_t> active_warp() const {
        if (!processing_.current()) return std::nullopt;
        return current_entry_.current().warp_id;
    }
    bool is_coalesced() const { return is_coalesced_.current(); }
    uint32_t serial_index() const { return serial_index_.current(); }
    const AddrGenFIFOEntry* current_entry() const {
        return processing_.current() ? &current_entry_.current() : nullptr;
    }

private:
    LdStUnit& ldst_;
    L1Cache& cache_;
    LoadGatherBufferFile& gather_file_;
    uint32_t line_size_;
    Stats& stats_;

    // Phase 6 of current_mut() elimination: durable cross-cycle coalescing
    // state, wrapped as Reg<T> for type-system enforcement of the
    // staged-write / latched-read discipline. evaluate() reads .next()
    // (the seeded staged value), mutates in place via next_mut() / set_next,
    // and commit_all() flips at the cycle boundary. Public accessors read
    // .current() so cross-module readers (tests, trace) see committed
    // state.
    Reg<bool> processing_;
    Reg<AddrGenFIFOEntry> current_entry_;
    Reg<bool> is_coalesced_;
    Reg<uint32_t> serial_index_;  // For serialized requests

    // Phase 6 of current_mut() elimination: cmd_in_flight_ tracks whether
    // a cmd was staged into cache last cycle (so this cycle's
    // cache.next_cmd_ready() is meaningful). Set at the bottom of evaluate
    // when we stage; reset at the top of evaluate after we read the ack.
    // Cross-cycle durable — read at top of next evaluate, so wrapped as
    // Reg<bool> (the audit candidate suggested Wire but the carry-across-
    // cycle semantics are register, not wire).
    Reg<bool> cmd_in_flight_;

    // Phase M1: pop intent staged here at evaluate; applied at commit by
    // calling ldst_.pop_front(). Producer (LdStUnit) only writes the back
    // of the deque at its own commit, so the front this consumer reads at
    // commit is identical to the front it observed at evaluate — the
    // `current_fifo_empty()` defensive check below should never fire.
    // Phase 7 candidate: Wire<bool> (evaluate→commit handoff).
    bool next_pop_ = false;  // scratch
};

} // namespace gpu_sim
