#pragma once

#include "gpu_sim/timing/pipeline_stage.h"
#include "gpu_sim/timing/reg.h"
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

class OperandCollector : public PipelineStage, public RegisteredStage {
public:
    explicit OperandCollector(Stats& stats) : stats_(stats) {
        register_state(&busy_, &cycles_remaining_, &instr_, &output_);
    }

    // Back-pressure discipline (REGISTERED + back-pressure direction):
    // current_busy() is a const accessor reading only committed
    // (current_busy_) state.
    //
    // Phase 10B.0: the WarpScheduler no longer reads this for issue gating —
    // operand-collector availability is now predicted scheduler-side via the
    // opcoll_cooldown_cycles_ countdown. current_busy() is retained for the
    // panic-drain query (TimingModel::execution_units_drained) and post-commit
    // observers (pipeline_drained / trace_cycle / unit tests).
    bool current_busy() const { return busy_.current(); }

    // Phase 10B.0.5 / Phase 4 (reg.h migration): explicit double-buffering via
    // seed_all(). Pre-Phase-4 the hand-rolled seed_next() copied busy_ /
    // cycles_remaining_ / instr_ but NOT output_ (evaluate() resets output_
    // to nullopt at its top); seed_all() now seeds output_ as well, which is
    // byte-identical because evaluate() unconditionally re-stages output_ at
    // its top (`output_.set_next(std::nullopt);`) before any conditional write.
    void seed_next() { seed_all(); }
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
    const std::optional<DispatchInput>& current_output() const { return output_.current(); }

    // Read by build_cycle_snapshot() / record_cycle_trace(), which run after
    // every stage's commit(). They observe committed end-of-cycle state.
    bool busy() const { return busy_.current(); }
    uint32_t current_cycles_remaining() const { return cycles_remaining_.current(); }
    std::optional<uint32_t> resident_warp() const {
        if (!busy_.current()) return std::nullopt;
        return instr_.current().warp_id;
    }
    const IssueOutput* current_instruction() const {
        return busy_.current() ? &instr_.current() : nullptr;
    }

private:
    Stats& stats_;                       // config (back-pointer)

    // Phase 4 (reg.h migration): every cross-cycle field is a Reg<T>. busy_,
    // cycles_remaining_, and instr_ are genuine multi-cycle carry-forward
    // (a VDOT8 stays in opcoll for two cycles). output_ is the REGISTERED
    // opcoll->unit slot: evaluate() unconditionally re-stages it at the top
    // (`set_next(std::nullopt)`), so seeding it is byte-identical.
    Reg<bool> busy_;
    Reg<uint32_t> cycles_remaining_;
    Reg<IssueOutput> instr_;
    Reg<std::optional<DispatchInput>> output_;

    // Phase 10B.0.5: per-cycle scratch flag for Stats relocation. evaluate()
    // assigns it fresh (the busy condition); commit() consumes it to
    // increment operand_collector_busy_cycles, so a re-evaluated stalled cycle
    // does not double-count.
    // Phase 7 of current_mut() elimination: per-cycle scratch flag as
    // Wire<bool>. evaluate() drives the busy condition; commit() reads
    // .value() to increment operand_collector_busy_cycles.
    Wire<bool> busy_this_cycle_;

    // Phase 10B.1/10B.2 back-pointers. nullptr-tolerant for unit tests.
    WarpScheduler* scheduler_ = nullptr;     // back-pointer
    WritebackArbiter* wb_arbiter_ = nullptr; // back-pointer
};

} // namespace gpu_sim
