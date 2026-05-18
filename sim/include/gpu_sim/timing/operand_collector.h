#pragma once

#include "gpu_sim/timing/pipeline_stage.h"
#include "gpu_sim/timing/warp_scheduler.h"
#include "gpu_sim/timing/branch_shadow_tracker.h"
#include "gpu_sim/stats.h"
#include <optional>

namespace gpu_sim {

class WritebackArbiter;

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

    // Back-pressure discipline (REGISTERED + back-pressure direction):
    // current_busy() is a const accessor reading only committed
    // (current_busy_) state.
    //
    // Phase 10B.0: the WarpScheduler no longer reads this for issue gating —
    // operand-collector availability is now predicted scheduler-side via the
    // opcoll_cooldown_cycles_ countdown. current_busy() is retained for the
    // panic-drain query (TimingModel::execution_units_drained) and post-commit
    // observers (pipeline_drained / trace_cycle / unit tests).
    bool current_busy() const { return current_busy_; }

    // Phase 10B.0.5: explicit double-buffering. seed_next() copies the
    // carry-forward fields current_* -> next_* at the top of the tick.
    // busy_/cycles_remaining_/instr_ are genuine carry-forward — evaluate()
    // decrements cycles_remaining and a VDOT8's instr_ is still being
    // collected on its second opcoll cycle (the scheduler has stopped
    // presenting it by then). next_output_ is NOT seeded — evaluate() sets it
    // to nullopt at its top and recomputes it from scratch. Called at the top
    // of TimingModel::tick() alongside the units' seed_next().
    void seed_next();
    void evaluate() override;
    void commit() override;
    void reset() override;

    // Panic flush hook. Called from TimingModel::tick() at the commit-phase
    // boundary when the panic signal becomes active. Delegates to reset().
    void flush();

    // Phase 2 discipline: writes only into next_* slots. evaluate() runs
    // afterward in the same tick and consumes next_* (combinational forward
    // inside this stage). Public so isolated unit tests drive it directly.
    void accept(const IssueOutput& issue);

    // Phase 10B.1/10B.2 back-pointers. The opcoll pulls the scheduler's
    // committed output in evaluate() (REGISTERED scheduler->opcoll edge) and
    // reads the arbiter's writeback stall in commit(). Wired by TimingModel;
    // null in isolated unit tests (then accept() is driven directly and
    // commit() never stalls).
    void set_scheduler(WarpScheduler* scheduler) { scheduler_ = scheduler; }
    void set_writeback_arbiter(class WritebackArbiter* arbiter) {
        wb_arbiter_ = arbiter;
    }

    // Phase 10B.1: REGISTERED opcoll->unit output. current_output() returns
    // the committed slot; each execution unit pulls it at the top of its own
    // evaluate() and self-selects by decoded.target_unit. evaluate() continues
    // to write next_output_.
    const std::optional<DispatchInput>& current_output() const { return current_output_; }

    // Read by build_cycle_snapshot() / record_cycle_trace(), which run after
    // every stage's commit(). They observe committed end-of-cycle state.
    bool busy() const { return current_busy_; }
    uint32_t current_cycles_remaining() const { return current_cycles_remaining_; }
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

    // Phase 10B.0.5: per-cycle scratch flag for Stats relocation. evaluate()
    // assigns it fresh (the busy condition); commit() consumes it to
    // increment operand_collector_busy_cycles, so a re-evaluated stalled cycle
    // does not double-count.
    bool busy_this_cycle_ = false;

    // Phase 10B.1/10B.2 back-pointers. nullptr-tolerant for unit tests.
    WarpScheduler* scheduler_ = nullptr;
    WritebackArbiter* wb_arbiter_ = nullptr;
};

} // namespace gpu_sim
