#pragma once

#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/timing/operand_collector.h"
#include "gpu_sim/timing/reg.h"
#include "gpu_sim/stats.h"
#include <deque>

namespace gpu_sim {

class WritebackArbiter;

class MultiplyUnit : public ExecutionUnit, public RegisteredStage {
public:
    struct PipelineSnapshot {
        uint32_t warp_id = 0;
        uint32_t pc = 0;
        uint32_t raw_instruction = 0;
        uint8_t dest_reg = 0;
    };

    MultiplyUnit(uint32_t pipeline_stages, Stats& stats)
        : pipeline_stages_(pipeline_stages), stats_(stats) {
        register_state(&pipeline_, &result_buffer_);
    }

    bool current_busy() const override {
        // Phase 10B.3: the scheduler no longer polls this — MULTIPLY is fully
        // pipelined (kUnitIterationLatency 0) and issue-gated by the writeback
        // bitmap alone. current_busy() now serves only the panic-drain query
        // (execution_units_drained / pipeline_drained), which needs "is any
        // work still in flight": committed pipeline non-empty or a committed
        // result not yet retired.
        return !pipeline_.current().empty() || result_buffer_.current().valid;
    }
    // Phase 10B.1/10B.3 back-pointers. nullptr-tolerant for unit tests.
    void set_operand_collector(class OperandCollector* opcoll) {
        opcoll_ = opcoll;
    }
    void set_writeback_arbiter(class WritebackArbiter* arbiter) {
        wb_arbiter_ = arbiter;
    }
    void set_sim_cycle(const uint64_t* cycle) { sim_cycle_ = cycle; }

    // Phase 10B.0.5: re-establish the carry-forward pipeline state at the top
    // of the tick. evaluate() consumes the prior-cycle pipeline state
    // (decrements cycles_remaining, shifts the deque), so the pipeline_ deque
    // is genuine multi-cycle carry-forward. seed_all() also seeds result_buffer_;
    // that is byte-identical because evaluate() unconditionally reassigns it
    // (WritebackEntry{}) at the top before any conditional write — see
    // multiply_unit.cpp::evaluate(). (The pre-Phase-3 header comment that the
    // result buffer was "or held" was stale; there is no held branch.)
    void seed_next() override { seed_all(); }
    void evaluate() override;
    void commit() override;
    void reset() override { reset_all(); busy_this_cycle_ = false; accepted_this_cycle_ = false; }
    bool current_has_result() const override;
    WritebackEntry consume_result() override;
    ExecUnit get_type() const override { return ExecUnit::MULTIPLY; }

    void accept(const DispatchInput& input, uint64_t cycle);
    bool busy() const { return !pipeline_.current().empty(); }
    uint32_t pipeline_occupancy() const { return static_cast<uint32_t>(pipeline_.current().size()); }
    std::vector<uint32_t> active_warps() const;
    std::vector<PipelineSnapshot> pipeline_snapshot() const;
    const WritebackEntry* result_entry() const {
        // Phase 10B.3: the result buffer is a plain double-buffered pipeline
        // register. The trace path runs after commit(), so it reads the
        // committed slot.
        return result_buffer_.current().valid ? &result_buffer_.current() : nullptr;
    }

private:
    struct PipelineEntry {
        WritebackEntry wb;
        uint32_t cycles_remaining;
    };

    uint32_t pipeline_stages_;
    Stats& stats_;
    // Phase 3 (reg.h migration): the multiply pipeline and result buffer are
    // both Reg<T>. accept() / evaluate() write only the staged slot; commit()
    // latches. Phase 10B.3: the result buffer is a plain double-buffered
    // pipeline register — current_has_result() and result_entry() read the
    // committed slot (REGISTERED unit->arbiter edge); consume_result() is a
    // pure read. current_busy() reads the committed slot (the panic-drain
    // query).
    Reg<std::deque<PipelineEntry>> pipeline_;
    Reg<WritebackEntry> result_buffer_;

    // Phase 10B.0.5: per-cycle scratch flags for Stats relocation. evaluate()
    // assigns busy_this_cycle_ fresh (the pipeline-non-empty condition);
    // accept() sets accepted_this_cycle_. Both are consumed at commit() to
    // perform the mul_stats increments, so a re-evaluated stalled cycle does
    // not double-count. commit() clears accepted_this_cycle_ (accept() may
    // not run every cycle); busy_this_cycle_ is assigned fresh by evaluate().
    bool busy_this_cycle_ = false;       // scratch
    bool accepted_this_cycle_ = false;   // scratch

    // Phase 10B.1/10B.3 back-pointers. nullptr-tolerant for unit tests.
    OperandCollector* opcoll_ = nullptr;
    WritebackArbiter* wb_arbiter_ = nullptr;
    const uint64_t* sim_cycle_ = nullptr;
};

} // namespace gpu_sim
