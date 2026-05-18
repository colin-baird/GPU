#pragma once

#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/timing/operand_collector.h"
#include "gpu_sim/timing/branch_predictor.h"
#include "gpu_sim/timing/branch_shadow_tracker.h"
#include "gpu_sim/stats.h"
#include <optional>

namespace gpu_sim {

class ALUUnit : public ExecutionUnit {
public:
    explicit ALUUnit(Stats& stats) : stats_(stats) {}

    bool current_busy() const override {
        return current_result_buffer_.valid || current_has_pending_;
    }

    // Phase 10B.0 interim issue gate (DELIBERATE, HUMAN-APPROVED DEVIATION
    // from the plan — to be REMOVED in Phase 10B.3). The plan's 10B.0 removes
    // the scheduler's current_busy() poll, replacing the structural-input
    // hazard with scheduler-side countdowns + the writeback bitmap. But the
    // old current_busy() was doing double duty: it also reported "result
    // buffer still occupied (not yet consumed by the arbiter)". 10B.0's
    // bitmap only prevents fixed-vs-fixed writeback collisions at issue; the
    // arbiter is still round-robin and does not honor the bitmap until
    // 10B.3. A load preempting this unit's fixed-latency writeback leaves an
    // unconsumed result in next_result_buffer_ that this unit's next-cycle
    // evaluate() would overwrite -> lost writeback -> scoreboard destination
    // never cleared -> dependent warps deadlock. current_result_pending() is
    // the narrow current_result_buffer_.valid portion of the old
    // current_busy(); the scheduler reads it as a backward committed-state
    // back-pressure read and skips issuing into a unit whose result buffer is
    // still occupied. Phase 10B.3's fixed-priority arbiter + combinational-
    // backward writeback stall subsume this gate, at which point it is
    // removed. (Phase 10F's doc sweep records the deviation.)
    bool current_result_pending() const { return current_result_buffer_.valid; }

    // Phase 10B.0.5: the ALU has 1-cycle latency, so its execution slot
    // (has_pending_ / pending_input_) is a per-cycle latch of opcoll's
    // output — not multi-cycle carry-forward state. evaluate() consumes the
    // value written by accept() THIS tick, never a prior committed value, so
    // there is nothing to re-establish in next_*. seed_next() is therefore
    // empty (the ExecutionUnit interface still carries it for the iterative
    // units). See the classification criterion in the phase-10 plan.
    void seed_next() override {}
    void evaluate() override;
    void commit() override;
    void reset() override;
    bool next_has_result() const override;
    WritebackEntry consume_result() override;
    ExecUnit get_type() const override { return ExecUnit::ALU; }

    // Phase 10A: branch resolution moved here from TimingModel::tick(). The
    // ALU resolves a branch in its own evaluate() — it updates the branch
    // predictor, and on misprediction publishes a REGISTERED redirect into
    // next_redirect_request_; on a correct prediction it clears the
    // branch-shadow tracker's in-flight bit. Wired by TimingModel after
    // construction (nullptr-tolerant for unit tests in isolation).
    void set_branch_tracker(BranchShadowTracker* tracker) {
        branch_tracker_ = tracker;
    }
    void set_branch_predictor(BranchPredictor* predictor) {
        branch_predictor_ = predictor;
    }

    // Phase 10A REGISTERED redirect-request signal. evaluate() writes
    // next_redirect_request_ on a mispredicted branch; commit() flips it to
    // current_redirect_request_. FetchStage and DecodeStage read
    // current_redirect_request() during their own commit() to apply the
    // flush. Kept REGISTERED as an interim staging step (see RedirectRequest
    // comment in execution_unit.h); converted to combinational backward in
    // Phase 10E.
    const RedirectRequest& current_redirect_request() const {
        return current_redirect_request_;
    }

    // Folds the override-vs-current_redirect_request_ choice into a member
    // function so the libclang AST extractor sees a statically resolvable
    // receiver. FetchStage::commit() and DecodeStage::commit() call this
    // through their wired `alu_` pointer; the caller still gates on
    // `alu_ != nullptr` for the unit-test default.
    RedirectRequest current_redirect_request_or_override(
        const std::optional<RedirectRequest>& override_request) const {
        if (override_request) return *override_request;
        return current_redirect_request_;
    }

    // Test hook: explicit override of the redirect-request signal, used by
    // tests that drive ALUUnit in isolation. Mirrors the prior opcoll hook.
    void set_redirect_request_override(bool valid, uint32_t warp_id,
                                       uint32_t target_pc) {
        RedirectRequest req;
        req.valid = valid;
        req.warp_id = warp_id;
        req.target_pc = target_pc;
        next_redirect_request_ = req;
    }

    // Phase 10A: shared misprediction check, moved from TimingModel. A branch
    // is mispredicted when the predicted next-PC differs from the resolved
    // actual next-PC. Public + static so the trace path (branch_redirect
    // event emission) can reuse it without an ALU instance.
    static bool branch_mispredicted(const DispatchInput& input);

    void accept(const DispatchInput& input, uint64_t cycle);
    bool busy() const { return current_has_pending_; }
    std::optional<uint32_t> active_warp() const {
        if (!current_has_pending_) return std::nullopt;
        return current_pending_input_.warp_id;
    }
    const DispatchInput* pending_input() const {
        return current_has_pending_ ? &current_pending_input_ : nullptr;
    }
    const WritebackEntry* result_entry() const {
        // Matches next_has_result(): read next_* so same-tick popped results are
        // visible to the writeback arbiter and the post-evaluate trace path.
        return next_result_buffer_.valid ? &next_result_buffer_ : nullptr;
    }

private:
    Stats& stats_;
    // Double-buffered cross-cycle state. accept() / evaluate() / consume_result()
    // write only next_*; commit() flips next_* -> current_*. External readers
    // (writeback arbiter, scheduler, panic drain, snapshot) see current_*.
    WritebackEntry current_result_buffer_;
    WritebackEntry next_result_buffer_;
    bool current_has_pending_ = false;
    bool next_has_pending_ = false;
    DispatchInput current_pending_input_;
    DispatchInput next_pending_input_;
    // Phase 10B.0.5: not double-buffered. accept() writes it and evaluate()
    // reads it within the same tick (into next_result_buffer_.issue_cycle);
    // no consumer ever reads a prior committed value, so it carries no
    // cross-cycle information for a 1-cycle ALU and needs no current_/next_
    // pair.
    uint64_t pending_cycle_ = 0;

    // Phase 10B.0.5: per-cycle scratch flag. evaluate() sets it to whether it
    // processed an instruction this cycle; commit() consumes it to relocate
    // the alu_stats.busy_cycles / .instructions increments out of evaluate()
    // (a re-evaluated stalled cycle must not double-count Stats artifacts).
    // Not a double-buffered field — evaluate() assigns it fresh each cycle.
    bool processed_this_cycle_ = false;

    // Phase 10A REGISTERED redirect-request signal. evaluate() writes
    // next_redirect_request_ on a mispredicted branch; commit() flips it to
    // current_redirect_request_. FetchStage / DecodeStage read
    // current_redirect_request() during their own commit().
    RedirectRequest current_redirect_request_{};
    RedirectRequest next_redirect_request_{};

    // Wired post-construction; nullptr-tolerant for unit tests in isolation.
    BranchShadowTracker* branch_tracker_ = nullptr;
    BranchPredictor* branch_predictor_ = nullptr;
};

} // namespace gpu_sim
