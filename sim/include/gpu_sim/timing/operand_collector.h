#pragma once

#include "gpu_sim/timing/pipeline_stage.h"
#include "gpu_sim/timing/warp_scheduler.h"
#include "gpu_sim/timing/branch_shadow_tracker.h"
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

// Phase 5 REGISTERED branch-redirect signal. The OperandCollector publishes
// next_redirect_request_ when a branch resolves with misprediction during
// its evaluate(); commit() flips next_-> current_; FetchStage and
// DecodeStage read current_redirect_request() during their own commit() and
// flush as needed. This replaces the prior mid-tick fetch_->redirect_warp
// and decode_->invalidate_warp side-channel calls from timing_model.cpp.
struct RedirectRequest {
    bool valid = false;
    uint32_t warp_id = 0;
    uint32_t target_pc = 0;
};

class OperandCollector : public PipelineStage {
public:
    explicit OperandCollector(Stats& stats) : stats_(stats) {}

    // Phase 5 wiring: opcoll knows about the branch-shadow tracker so
    // resolve_branch() can clear it directly into next_*. Wired by
    // TimingModel after construction (nullptr-tolerant for unit tests
    // that exercise OperandCollector in isolation).
    void set_branch_tracker(BranchShadowTracker* tracker) {
        branch_tracker_ = tracker;
    }

    // Phase 5 REGISTERED branch resolution: called from TimingModel::tick()
    // after opcoll_->evaluate() has produced its output for a BRANCH/JAL/JALR.
    // Always clears branch_in_flight in next_ for the resolving warp; if
    // mispredicted == true, also writes next_redirect_request_ so that
    // fetch.commit() and decode.commit() can flush this same cycle's
    // boundary. The redirect itself is delayed by 1 cycle relative to the
    // pre-Phase-5 mid-tick fetch_->redirect_warp call: fetch/decode see
    // current_redirect_request_ in cycle N+1 after commit() flips next_.
    void resolve_branch(uint32_t warp_id, bool mispredicted, uint32_t target_pc);
    const RedirectRequest& current_redirect_request() const {
        return current_redirect_request_;
    }

    // Phase 4 READY/STALL discipline: compute_ready() reads only committed
    // (current_busy_) state and updates ready_out_. WarpScheduler::evaluate()
    // queries ready_out() during the same tick; TimingModel::tick() invokes
    // compute_ready() before scheduler_->evaluate() so the signal is stable
    // and derived from end-of-last-cycle state. Phase 8: overrides
    // PipelineStage's virtual default no-op.
    void compute_ready() override;
    bool ready_out() const { return ready_out_; }

    void evaluate() override;
    void commit() override;
    void reset() override;

    // Phase 6: panic flush hook. Called from TimingModel::tick() at the
    // commit-phase boundary when the panic signal becomes active. Same
    // body as reset() — drops the in-flight instruction, clears
    // busy/cycles, and clears the redirect-request slot.
    void flush();

    // Alias of ready_out() retained for post-commit observers and tests:
    // pipeline_drained() / execution_units_drained() / trace_cycle() / unit
    // tests query "is operand collector idle?" after commit and need to read
    // committed (current_busy_) state directly. Functionally identical to
    // ready_out() because both derive from current_busy_.
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

    // Phase 4 READY/STALL slot: written by compute_ready() at the top of the
    // tick from current_busy_; read by WarpScheduler::evaluate() the same
    // cycle. Initial value matches pre-tick "free" so the very first cycle
    // matches the prior pre-evaluate setter behavior.
    bool ready_out_ = true;

    // Phase 5 REGISTERED redirect-request signal. resolve_branch() writes
    // next_redirect_request_; commit() flips it to current_redirect_request_.
    // FetchStage and DecodeStage read current_redirect_request() during
    // their own commit() to apply the flush, replacing the prior mid-tick
    // side-channel calls from timing_model.cpp.
    RedirectRequest current_redirect_request_{};
    RedirectRequest next_redirect_request_{};

    BranchShadowTracker* branch_tracker_ = nullptr;
};

} // namespace gpu_sim
