#pragma once

#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/timing/operand_collector.h"
#include "gpu_sim/stats.h"

namespace gpu_sim {

class WritebackArbiter;

class DivideUnit : public ExecutionUnit {
public:
    explicit DivideUnit(Stats& stats) : stats_(stats) {}

    bool current_busy() const override {
        return current_busy_ || current_result_buffer_.valid;
    }

    // Phase 10B.1/10B.3 back-pointers. nullptr-tolerant for unit tests.
    void set_operand_collector(class OperandCollector* opcoll) {
        opcoll_ = opcoll;
    }
    void set_writeback_arbiter(class WritebackArbiter* arbiter) {
        wb_arbiter_ = arbiter;
    }
    void set_sim_cycle(const uint64_t* cycle) { sim_cycle_ = cycle; }

    // Phase 10B.0.5: copy the carry-forward iterative state current_* ->
    // next_*. evaluate() consumes the prior-cycle busy flag and decrements
    // cycles_remaining, so busy_/cycles_remaining_/pending_result_ are genuine
    // multi-cycle carry-forward. The result buffer is NOT seeded — evaluate()
    // assigns it fresh when the countdown reaches zero.
    void seed_next() override;
    void evaluate() override;
    void commit() override;
    void reset() override;
    bool current_has_result() const override;
    WritebackEntry consume_result() override;
    ExecUnit get_type() const override { return ExecUnit::DIVIDE; }

    void accept(const DispatchInput& input, uint64_t cycle);
    bool busy() const { return current_busy_; }
    uint32_t current_cycles_remaining() const { return current_cycles_remaining_; }
    std::optional<uint32_t> active_warp() const {
        if (!current_busy_) return std::nullopt;
        return current_pending_result_.warp_id;
    }
    const WritebackEntry* pending_entry() const {
        return current_busy_ ? &current_pending_result_ : nullptr;
    }
    const WritebackEntry* result_entry() const {
        // Phase 10B.3: the result buffer is a plain double-buffered pipeline
        // register. The trace path runs after commit(), so it reads current_*.
        return current_result_buffer_.valid ? &current_result_buffer_ : nullptr;
    }

private:
    // Phase 10B.0: the latency lives in execution_unit.h as kDivideLatency
    // (single source of truth, also consumed by the scheduler issue gate).
    static constexpr uint32_t DIVIDE_LATENCY = kDivideLatency;

    Stats& stats_;
    // Phase 1 discipline: every cross-cycle field is double-buffered.
    bool current_busy_ = false;
    bool next_busy_ = false;
    uint32_t current_cycles_remaining_ = 0;
    uint32_t next_cycles_remaining_ = 0;
    WritebackEntry current_pending_result_;
    WritebackEntry next_pending_result_;
    WritebackEntry current_result_buffer_;
    WritebackEntry next_result_buffer_;

    // Phase 10B.0.5: per-cycle scratch flags for Stats relocation. evaluate()
    // assigns busy_this_cycle_ fresh; accept() sets accepted_this_cycle_. Both
    // consumed at commit() so a re-evaluated stalled cycle does not
    // double-count div_stats.
    bool busy_this_cycle_ = false;
    bool accepted_this_cycle_ = false;

    // Phase 10B.1/10B.3 back-pointers. nullptr-tolerant for unit tests.
    OperandCollector* opcoll_ = nullptr;
    WritebackArbiter* wb_arbiter_ = nullptr;
    const uint64_t* sim_cycle_ = nullptr;
};

} // namespace gpu_sim
