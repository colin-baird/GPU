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
    // are genuine carry-forward. seed_all() re-establishes them at tick top.
    // The addr-gen FIFO itself lives on TimingModel (a cross-stage RegFifo
    // peer of LdStUnit / CoalescingUnit) and is committed in TimingModel's
    // dedicated ungated cross-stage FIFO commit pass, so it is not part of
    // this stage's RegisteredStage list.
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
    // Phase 3 (close-the-Reg-family-migration): cross-stage addr-gen FIFO
    // back-pointer. The FIFO is declared as a direct member of TimingModel
    // (a peer of this stage and the consumer CoalescingUnit, not a member of
    // either); TimingModel hands a back-pointer to both stages at
    // construction. Committed in TimingModel's dedicated ungated cross-stage
    // FIFO commit pass — NEVER enrolled in this stage's gated commit_all().
    // nullptr-tolerant for unit tests that exercise the stage in isolation
    // (those tests use the unit-test-only addr_gen_fifo_test_only_ storage
    // path below).
    void set_addr_gen_fifo(RegFifo<AddrGenFIFOEntry>* fifo) {
        addr_gen_fifo_ = fifo;
    }

    // Phase 3 (close-the-Reg-family-migration): cross-stage addr-gen FIFO
    // accessors. The FIFO lives on TimingModel and is committed in its
    // dedicated ungated cross-stage commit pass. LdStUnit::evaluate() stages
    // the push gated on !wb_arbiter_->next_writeback_stall() (the literal
    // simulator translation of the RTL wr_en && !stall AND-gate);
    // CoalescingUnit::evaluate() stages the pop unconditionally on its pop
    // decision. Both intents apply atomically pop-then-push at the
    // cross-stage commit pass. On a stalled cycle: producer's pipeline
    // registers freeze (LdStUnit's own commit_all() is gated and early-
    // returns), so its push is naturally re-staged on the resumed cycle;
    // the consumer's pop still applies, draining the FIFO normally.
    //
    // The accessors below are kept as delegating reads of the back-pointed
    // FIFO for compatibility with the many external callers (scheduler,
    // snapshot builder, drained queries, unit tests).
    bool current_fifo_empty() const {
        return addr_gen_fifo_ ? addr_gen_fifo_->current_empty() : true;
    }
    const AddrGenFIFOEntry& current_fifo_front() const {
        return addr_gen_fifo_->current_front();
    }
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
    const std::deque<AddrGenFIFOEntry>& current_fifo_entries() const {
        return addr_gen_fifo_ ? addr_gen_fifo_->current() : empty_fifo_view_;
    }
    // Size accessor for upstream issue-gate bookkeeping (consumed by the
    // pipeline plan's Phase 10B.0 LDST FIFO accounting). REGISTERED: reads the
    // committed addr-gen FIFO.
    uint32_t current_fifo_size() const {
        return addr_gen_fifo_
            ? static_cast<uint32_t>(addr_gen_fifo_->current_size())
            : 0u;
    }

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
    // Reg<T>. The addr-gen FIFO itself is a RegFifo declared on TimingModel
    // and reached via the addr_gen_fifo_ back-pointer below (cross-stage
    // ownership pattern — see public-section comment above).
    Reg<bool> busy_;
    Reg<uint32_t> cycles_remaining_;
    Reg<AddrGenFIFOEntry> pending_entry_;
    // Phase 3 (close-the-Reg-family-migration): back-pointer to the cross-
    // stage addr-gen FIFO. The actual RegFifo lives on TimingModel as a peer
    // of LdStUnit / CoalescingUnit and is committed in TimingModel's
    // dedicated ungated cross-stage FIFO commit pass. Wired here via
    // set_addr_gen_fifo() at construction. nullptr-tolerant for unit tests
    // that exercise this stage in isolation. back-pointer
    RegFifo<AddrGenFIFOEntry>* addr_gen_fifo_ = nullptr;  // timing-naming-allow: back-pointer to TimingModel-owned cross-stage RegFifo; the FIFO itself is enrolled and committed at the TimingModel-owned cross-stage commit pass.
    // Empty fallback view returned by current_fifo_entries() when the back-
    // pointer is null (unit-test paths). const-after-construction sentinel —
    // not state, just a return-by-reference target for the nullptr branch.
    const std::deque<AddrGenFIFOEntry> empty_fifo_view_{};  // config
    // Phase 10B.0: monotonic push counter. Incremented in commit() on the
    // cycle the staged push lands in the cross-stage addr-gen FIFO — the same
    // cycle the op becomes visible to consumers via current_fifo_size(). The
    // gating is consistent: LdStUnit::evaluate() conditions stage_push on
    // !writeback_stall (the RTL wr_en mask), and LdStUnit::commit() is itself
    // gated on the same signal, so push_staged_this_cycle_ is driven iff the
    // FIFO will receive the push at the cross-stage commit pass that runs
    // after this stage's commit. Plain uint32_t: monotonic sim-instrumentation
    // accumulator, not a clocked-hardware register.
    uint32_t fifo_total_pushes_ = 0;     // sim-instrumentation

    // Phase 10B.0.5: per-cycle scratch flags for Stats relocation. evaluate()
    // assigns busy_this_cycle_ fresh; accept() sets accepted_this_cycle_. Both
    // consumed at commit() so a re-evaluated stalled cycle does not
    // double-count ldst_stats.
    // Phase 7 of current_mut() elimination: per-cycle scratch flags as Wire<bool>.
    Wire<bool> busy_this_cycle_;
    Wire<bool> accepted_this_cycle_;
    // Phase 3 (close-the-Reg-family-migration): per-cycle Wire tracking
    // whether evaluate() staged a push onto the cross-stage addr-gen FIFO.
    // Consumed at commit() to advance fifo_total_pushes_ in lockstep with
    // the push application (commit is gated on the same writeback stall as
    // the evaluate-time stage_push, so the gating is automatic — if the
    // stage's commit_all() doesn't run, the Wire's de-assert is irrelevant
    // because the counter increment is skipped along with the commit).
    Wire<bool> push_staged_this_cycle_;

    // Phase 10B.1/10B.3 back-pointers. nullptr-tolerant for unit tests.
    OperandCollector* opcoll_ = nullptr;
    WritebackArbiter* wb_arbiter_ = nullptr;
    const uint64_t* sim_cycle_ = nullptr;
};

} // namespace gpu_sim
