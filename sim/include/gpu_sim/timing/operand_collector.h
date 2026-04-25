#pragma once

#include "gpu_sim/timing/pipeline_stage.h"
#include "gpu_sim/timing/warp_scheduler.h"
#include "gpu_sim/stats.h"
#include <optional>

namespace gpu_sim {

struct DispatchInput {
    DecodedInstruction decoded;
    TraceEvent trace;
    uint32_t warp_id;
    uint32_t pc;
    BranchPrediction prediction;
};

class OperandCollector : public PipelineStage {
public:
    explicit OperandCollector(Stats& stats) : stats_(stats) {}

    void evaluate() override;
    void commit() override;
    void reset() override;

    // Queried by the scheduler BEFORE this tick's accept()/evaluate(). Reads
    // committed (current_*) state — last cycle's end-of-cycle busy bit. The
    // scheduler's pre-evaluate read at timing_model.cpp ~line 390 must see
    // the same value the cycle would have shown under the prior single-buffer
    // implementation.
    bool is_free() const { return !current_busy_; }

    // Phase 2 discipline: writes only into next_* slots. evaluate() runs
    // afterward in the same tick and consumes next_* (combinational forward
    // inside this stage).
    void accept(const IssueOutput& issue);

    // Output to dispatch — COMBINATIONAL forward edge to dispatch_to_unit and
    // each execution unit's accept(). Returns next_output_ so the just-
    // produced output is visible to downstream units this same tick.
    std::optional<DispatchInput>& output() { return next_output_; }
    const std::optional<DispatchInput>& current_output() const { return current_output_; }

    // Read by build_cycle_snapshot() / record_cycle_trace(), which run after
    // every stage's commit(). They observe committed end-of-cycle state.
    bool busy() const { return current_busy_; }
    uint32_t cycles_remaining() const { return current_cycles_remaining_; }
    std::optional<uint32_t> resident_warp() const {
        if (!current_busy_) return std::nullopt;
        return current_instr_.warp_id;
    }
    const IssueOutput* current_instruction() const {
        return current_busy_ ? &current_instr_ : nullptr;
    }

private:
    Stats& stats_;

    // Cross-cycle (REGISTERED) state: busy bit, countdown, and the in-flight
    // IssueOutput. accept() writes only next_*; evaluate() consumes next_*
    // (which equals current_* after commit() unless accept() just overrode
    // it); commit() flips next_* -> current_*.
    bool current_busy_ = false;
    bool next_busy_ = false;
    uint32_t current_cycles_remaining_ = 0;
    uint32_t next_cycles_remaining_ = 0;
    IssueOutput current_instr_{};
    IssueOutput next_instr_{};

    // Same-tick COMBINATIONAL output payload. next_output_ is what evaluate()
    // produces and downstream stages read this cycle; current_output_ is the
    // committed copy used by post-commit observers (branch_redirect tracing).
    std::optional<DispatchInput> current_output_;
    std::optional<DispatchInput> next_output_;
};

} // namespace gpu_sim
