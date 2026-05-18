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
    uint64_t current_pending_cycle_ = 0;
    uint64_t next_pending_cycle_ = 0;

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
