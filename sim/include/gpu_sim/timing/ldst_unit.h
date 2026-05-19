#pragma once

#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/timing/operand_collector.h"
#include "gpu_sim/timing/reg.h"
#include "gpu_sim/stats.h"
#include <deque>
#include <optional>

namespace gpu_sim {

class WritebackArbiter;

struct AddrGenFIFOEntry {
    bool valid = false;
    uint32_t warp_id;
    uint8_t dest_reg;
    bool is_load;
    bool is_store;
    TraceEvent trace;  // Contains addresses, store data, etc.
    uint64_t issue_cycle;
};

class LdStUnit : public ExecutionUnit, public RegisteredStage {
public:
    LdStUnit(uint32_t num_ldst_units, uint32_t fifo_depth, Stats& stats);

    bool current_busy() const override { return busy_.current(); }
    // Phase 10B.0.5: LdSt is a full seed_next participant. Address generation
    // is multi-cycle — while an addr-gen op is in flight evaluate() consumes
    // busy_/cycles_remaining_/pending_entry_ from the prior cycle — so these
    // are genuine carry-forward. Phase 3: seed_all() re-establishes them at
    // tick top. addr_gen_fifo_ / next_push_ are plain deque + optional and not
    // part of the RegisteredStage list (see public-section comment below).
    void seed_next() override { seed_all(); }
    void evaluate() override;
    void commit() override;
    void reset() override;
    bool current_has_result() const override;
    WritebackEntry consume_result() override;
    ExecUnit get_type() const override { return ExecUnit::LDST; }

    void accept(const DispatchInput& input, uint64_t cycle);

    // Phase 10B.1/10B.3 back-pointers. nullptr-tolerant for unit tests.
    void set_operand_collector(class OperandCollector* opcoll) {
        opcoll_ = opcoll;
    }
    void set_writeback_arbiter(class WritebackArbiter* arbiter) {
        wb_arbiter_ = arbiter;
    }
    void set_sim_cycle(const uint64_t* cycle) { sim_cycle_ = cycle; }

    // Phase M1: REGISTERED address-gen FIFO. Mutated only at the commit
    // phase — producer (this unit) writes next_push_ at evaluate and applies
    // it at LdStUnit::commit (gated by the writeback stall); consumer
    // (coalescing) drives pop_front() from its own commit (ungated). The
    // asymmetric gating is deliberate: on a writeback-stall cycle, coalescing
    // may still drain the FIFO while this unit holds its push, and the held
    // entry re-pushes on the resumed cycle. Reads during evaluate see the
    // start-of-cycle state.
    //
    // This FIFO is intentionally hand-rolled rather than wrapped in a
    // RegFifo<T> — the simple RegFifo applies pop-then-push atomically inside
    // a single owning stage's commit(), which cannot express the cross-stage
    // commit-gating asymmetry above (Phase 3 verified this empirically: the
    // RegFifo migration produced cycle-count deltas across all six workload
    // benchmarks). The deque + next_push_ + caller-driven pop_front() pattern
    // remains the byte-identical encoding.
    bool current_fifo_empty() const { return addr_gen_fifo_.empty(); }
    const AddrGenFIFOEntry& current_fifo_front() const { return addr_gen_fifo_.front(); }
    void pop_front() { addr_gen_fifo_.pop_front(); }
    bool busy() const { return busy_.current(); }
    uint32_t current_cycles_remaining() const { return cycles_remaining_.current(); }
    std::optional<uint32_t> active_warp() const {
        if (!busy_.current()) return std::nullopt;
        return pending_entry_.current().warp_id;
    }
    const AddrGenFIFOEntry* pending_entry() const {
        return pending_entry_.current().valid ? &pending_entry_.current() : nullptr;
    }
    // FIFO view for trace dumps. Reads the stable committed deque (same state
    // observed by coalescing during evaluate this cycle).
    const std::deque<AddrGenFIFOEntry>& current_fifo_entries() const { return addr_gen_fifo_; }
    // Size accessor for upstream issue-gate bookkeeping (consumed by the
    // pipeline plan's Phase 10B.0 LDST FIFO accounting). REGISTERED: reads the
    // committed addr-gen FIFO.
    uint32_t current_fifo_size() const { return static_cast<uint32_t>(addr_gen_fifo_.size()); }

    // Phase 10B.0: the addr-gen FIFO's configured depth (SimConfig
    // addr_gen_fifo_depth). Static after construction; the scheduler's LDST
    // FIFO-occupancy gate uses it as the issue ceiling.
    uint32_t current_fifo_capacity() const { return fifo_depth_; }

    // Phase 10B.0: address-generation latency, ceil(WARP_SIZE / num_ldst_units)
    // cycles. The addr-gen stage holds exactly one op at a time (a single
    // busy_ flag — accept() unconditionally overwrites it), so it is an
    // iterative structural hazard distinct from the addr-gen FIFO. The
    // scheduler arms its unit_busy_[LDST] countdown with this value so two
    // LDST ops are never issued closely enough for the second to clobber the
    // first mid-address-generation. Static after construction.
    uint32_t current_addr_gen_latency() const {
        return (WARP_SIZE + num_ldst_units_ - 1) / num_ldst_units_;
    }

    // Phase 10B.0: REGISTERED monotonic count of ops ever pushed into the
    // addr-gen FIFO, incremented at the commit-phase push (the same event
    // current_fifo_size() reflects). The scheduler's LDST FIFO-occupancy gate
    // computes the in-transit population as
    // (ldst_issued_total_ - current_fifo_total_pushes()); see warp_scheduler.
    // This is a REGISTERED, committed-state back-pressure read by the
    // scheduler (upstream of LDST) — discipline-compliant.
    uint32_t current_fifo_total_pushes() const { return fifo_total_pushes_; }

private:
    uint32_t num_ldst_units_;
    uint32_t fifo_depth_;
    Stats& stats_;

    // Phase 3 (reg.h migration): the per-cycle iterative state is wrapped in
    // Reg<T>; the addr-gen FIFO is intentionally NOT a RegFifo (see the
    // public-section comment above).
    Reg<bool> busy_;
    Reg<uint32_t> cycles_remaining_;
    Reg<AddrGenFIFOEntry> pending_entry_;
    // Phase M1: REGISTERED address-gen FIFO. Mutated only at the commit phase
    // — push applied here (gated), pop applied by coalescing (ungated).
    std::deque<AddrGenFIFOEntry> addr_gen_fifo_;
    std::optional<AddrGenFIFOEntry> next_push_;
    // Phase 10B.0: monotonic push counter. Incremented in commit() on the
    // cycle the staged push is applied — the same cycle the op becomes
    // visible in addr_gen_fifo_. Never decremented; the scheduler's
    // FIFO-occupancy gate uses it as a difference against ldst_issued_total_.
    // Plain uint32_t: monotonic sim-instrumentation accumulator, not a
    // clocked-hardware register.
    uint32_t fifo_total_pushes_ = 0;     // sim-instrumentation

    // Phase 10B.0.5: per-cycle scratch flags for Stats relocation. evaluate()
    // assigns busy_this_cycle_ fresh; accept() sets accepted_this_cycle_. Both
    // consumed at commit() so a re-evaluated stalled cycle does not
    // double-count ldst_stats.
    bool busy_this_cycle_ = false;       // scratch
    bool accepted_this_cycle_ = false;   // scratch

    // Phase 10B.1/10B.3 back-pointers. nullptr-tolerant for unit tests.
    OperandCollector* opcoll_ = nullptr;
    WritebackArbiter* wb_arbiter_ = nullptr;
    const uint64_t* sim_cycle_ = nullptr;
};

} // namespace gpu_sim
