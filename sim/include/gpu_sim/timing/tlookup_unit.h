#pragma once

#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/timing/operand_collector.h"
#include "gpu_sim/timing/reg.h"
#include "gpu_sim/stats.h"

namespace gpu_sim {

class WritebackArbiter;

class TLookupUnit : public ExecutionUnit, public RegisteredStage {
public:
    explicit TLookupUnit(Stats& stats) : stats_(stats) {
        register_state(&busy_, &cycles_remaining_, &pending_result_,
                       &result_buffer_);
    }

    bool current_busy() const override {
        return busy_.current() || result_buffer_.current().valid;
    }

    // Phase 10B.1/10B.3 back-pointers. nullptr-tolerant for unit tests.
    void set_operand_collector(class OperandCollector* opcoll) {
        opcoll_ = opcoll;
    }
    void set_writeback_arbiter(class WritebackArbiter* arbiter) {
        wb_arbiter_ = arbiter;
    }
    void set_sim_cycle(const uint64_t* cycle) { sim_cycle_ = cycle; }

    // Phase 10B.0.5: re-establish the carry-forward iterative state at the top
    // of the tick. busy_, cycles_remaining_, pending_result_ are genuine
    // multi-cycle carry-forward. seed_all() also seeds result_buffer_; that is
    // byte-identical because evaluate() unconditionally restages it
    // (WritebackEntry{}) at its top before any conditional write.
    void seed_next() override { seed_all(); }
    void evaluate() override;
    void commit() override;
    void reset() override { reset_all(); busy_this_cycle_ = false; accepted_this_cycle_ = false; }
    bool current_has_result() const override;
    WritebackEntry consume_result() override;
    ExecUnit get_type() const override { return ExecUnit::TLOOKUP; }

    void accept(const DispatchInput& input, uint64_t cycle);
    bool busy() const { return busy_.current(); }
    uint32_t current_cycles_remaining() const { return cycles_remaining_.current(); }
    std::optional<uint32_t> active_warp() const {
        if (!busy_.current()) return std::nullopt;
        return pending_result_.current().warp_id;
    }
    const WritebackEntry* pending_entry() const {
        return busy_.current() ? &pending_result_.current() : nullptr;
    }
    const WritebackEntry* result_entry() const {
        // Phase 10B.3: the result buffer is a plain double-buffered pipeline
        // register. The trace path runs after commit(), so it reads the
        // committed slot.
        return result_buffer_.current().valid ? &result_buffer_.current() : nullptr;
    }

private:
    // Pipelined dual-port BRAM: 2 lanes/cycle, ceil(32/2) + 1 drain = 17 cycles.
    // Phase 10B.0: the value lives in execution_unit.h as kTlookupLatency
    // (single source of truth, also consumed by the scheduler issue gate).
    static constexpr uint32_t TLOOKUP_LATENCY = kTlookupLatency;

    Stats& stats_;
    // Phase 3 (reg.h migration): every cross-cycle field is a Reg<T>.
    Reg<bool> busy_;
    Reg<uint32_t> cycles_remaining_;
    Reg<WritebackEntry> pending_result_;
    Reg<WritebackEntry> result_buffer_;

    // Phase 10B.0.5: per-cycle scratch flags for Stats relocation. evaluate()
    // assigns busy_this_cycle_ fresh; accept() sets accepted_this_cycle_. Both
    // consumed at commit() so a re-evaluated stalled cycle does not
    // double-count tlookup_stats.
    bool busy_this_cycle_ = false;       // scratch
    bool accepted_this_cycle_ = false;   // scratch

    // Phase 10B.1/10B.3 back-pointers. nullptr-tolerant for unit tests.
    OperandCollector* opcoll_ = nullptr;
    WritebackArbiter* wb_arbiter_ = nullptr;
    const uint64_t* sim_cycle_ = nullptr;
};

} // namespace gpu_sim
