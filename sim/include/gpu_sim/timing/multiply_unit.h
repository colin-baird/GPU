#pragma once

#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/timing/operand_collector.h"
#include "gpu_sim/stats.h"
#include <deque>

namespace gpu_sim {

class WritebackArbiter;

class MultiplyUnit : public ExecutionUnit {
public:
    struct PipelineSnapshot {
        uint32_t warp_id = 0;
        uint32_t pc = 0;
        uint32_t raw_instruction = 0;
        uint8_t dest_reg = 0;
    };

    MultiplyUnit(uint32_t pipeline_stages, Stats& stats)
        : pipeline_stages_(pipeline_stages), stats_(stats) {}

    bool current_busy() const override {
        // Phase 10B.3: the scheduler no longer polls this — MULTIPLY is fully
        // pipelined (kUnitIterationLatency 0) and issue-gated by the writeback
        // bitmap alone. current_busy() now serves only the panic-drain query
        // (execution_units_drained / pipeline_drained), which needs "is any
        // work still in flight": committed pipeline non-empty or a committed
        // result not yet retired.
        return !current_pipeline_.empty() || current_result_buffer_.valid;
    }
    // Phase 10B.1/10B.3 back-pointers. nullptr-tolerant for unit tests.
    void set_operand_collector(class OperandCollector* opcoll) {
        opcoll_ = opcoll;
    }
    void set_writeback_arbiter(class WritebackArbiter* arbiter) {
        wb_arbiter_ = arbiter;
    }
    void set_sim_cycle(const uint64_t* cycle) { sim_cycle_ = cycle; }

    // Phase 10B.0.5: copy the pipeline deque current_* -> next_*. evaluate()
    // consumes the prior-cycle pipeline state (decrements cycles_remaining,
    // shifts the deque), so it is genuine multi-cycle carry-forward. The
    // result buffer is NOT seeded — evaluate() assigns it fresh (or holds it),
    // so it is a plain double-buffered pipeline register.
    void seed_next() override;
    void evaluate() override;
    void commit() override;
    void reset() override;
    bool current_has_result() const override;
    WritebackEntry consume_result() override;
    ExecUnit get_type() const override { return ExecUnit::MULTIPLY; }

    void accept(const DispatchInput& input, uint64_t cycle);
    bool busy() const { return !current_pipeline_.empty(); }
    uint32_t pipeline_occupancy() const { return static_cast<uint32_t>(current_pipeline_.size()); }
    std::vector<uint32_t> active_warps() const;
    std::vector<PipelineSnapshot> pipeline_snapshot() const;
    const WritebackEntry* result_entry() const {
        // Phase 10B.3: the result buffer is a plain double-buffered pipeline
        // register. The trace path runs after commit(), so it reads current_*.
        return current_result_buffer_.valid ? &current_result_buffer_ : nullptr;
    }

private:
    struct PipelineEntry {
        WritebackEntry wb;
        uint32_t cycles_remaining;
    };

    uint32_t pipeline_stages_;
    Stats& stats_;
    // Phase 1 discipline: pipeline_ and result_buffer_ are double-buffered.
    // accept() / evaluate() write only next_*; commit() flips next_* ->
    // current_*. Phase 10B.3: the result buffer is a plain double-buffered
    // pipeline register — current_has_result() and result_entry() read
    // current_* (REGISTERED unit->arbiter edge); consume_result() is a pure
    // read. current_busy() reads current_* (the panic-drain query).
    std::deque<PipelineEntry> current_pipeline_;
    std::deque<PipelineEntry> next_pipeline_;
    WritebackEntry current_result_buffer_;
    WritebackEntry next_result_buffer_;

    // Phase 10B.0.5: per-cycle scratch flags for Stats relocation. evaluate()
    // assigns busy_this_cycle_ fresh (the pipeline-non-empty condition);
    // accept() sets accepted_this_cycle_. Both are consumed at commit() to
    // perform the mul_stats increments, so a re-evaluated stalled cycle does
    // not double-count. commit() clears accepted_this_cycle_ (accept() may
    // not run every cycle); busy_this_cycle_ is assigned fresh by evaluate().
    bool busy_this_cycle_ = false;
    bool accepted_this_cycle_ = false;

    // Phase 10B.1/10B.3 back-pointers. nullptr-tolerant for unit tests.
    OperandCollector* opcoll_ = nullptr;
    WritebackArbiter* wb_arbiter_ = nullptr;
    const uint64_t* sim_cycle_ = nullptr;
};

} // namespace gpu_sim
