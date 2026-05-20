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

    // Phase 3 (close-the-Reg-family-migration): cross-stage addr-gen FIFO
    // back-pointer. The FIFO is declared as a direct member of TimingModel
    // (a peer of CoalescingUnit and the producer LdStUnit, not a member of
    // either); TimingModel hands a back-pointer to both stages at
    // construction. CoalescingUnit::evaluate() calls stage_pop() directly on
    // it (unconditionally — modulo its own pop decision), and the cross-
    // stage commit pass on TimingModel applies the pop. nullptr-tolerant for
    // unit tests that exercise this stage in isolation.
    void set_addr_gen_fifo(RegFifo<AddrGenFIFOEntry>* fifo) {
        addr_gen_fifo_ = fifo;
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

    // Phase 3 (close-the-Reg-family-migration): back-pointer to the cross-
    // stage addr-gen FIFO owned by TimingModel. evaluate() calls stage_pop()
    // directly; the pop applies at the ungated cross-stage commit pass.
    // nullptr-tolerant for unit tests. back-pointer
    RegFifo<AddrGenFIFOEntry>* addr_gen_fifo_ = nullptr;  // timing-naming-allow: back-pointer to TimingModel-owned cross-stage RegFifo; the FIFO itself is enrolled and committed at the TimingModel-owned cross-stage commit pass.
};

} // namespace gpu_sim
