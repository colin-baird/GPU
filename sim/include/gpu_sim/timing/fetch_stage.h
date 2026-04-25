#pragma once

#include "gpu_sim/timing/pipeline_stage.h"
#include "gpu_sim/timing/branch_predictor.h"
#include "gpu_sim/timing/warp_state.h"
#include "gpu_sim/functional/memory.h"
#include "gpu_sim/stats.h"
#include <optional>

namespace gpu_sim {

class DecodeStage;          // forward decl: fetch reads decode.ready_to_consume_fetch()
class OperandCollector;     // forward decl: fetch reads opcoll.current_redirect_request()
class BranchShadowTracker;  // forward decl: fetch clears in-flight on redirect apply

struct FetchOutput {
    uint32_t raw_instruction;
    uint32_t warp_id;
    uint32_t pc;
    BranchPrediction prediction;
};

class FetchStage : public PipelineStage {
public:
    FetchStage(uint32_t num_warps, WarpState* warps,
               const InstructionMemory& imem, BranchPredictor& predictor, Stats& stats);

    void evaluate() override;
    void commit() override;
    void reset() override;
    std::optional<FetchOutput>& output() { return next_output_; }
    const std::optional<FetchOutput>& current_output() const { return current_output_; }

    // Wire decode after both stages are constructed. Fetch reads
    // decode->ready_to_consume_fetch() and decode->pending_warp() during
    // evaluate() as READY/STALL signals (Phase 3 discipline).
    void set_decode(const DecodeStage* decode) { decode_ = decode; }

    // Phase 5: wire opcoll so fetch.commit() can read its REGISTERED
    // current_redirect_request() and apply the flush from there. Replaces
    // the prior mid-tick fetch_->redirect_warp(...) side-channel call from
    // timing_model.cpp.
    void set_opcoll(const OperandCollector* opcoll) { opcoll_ = opcoll; }

    // Phase 5: wire branch-shadow tracker so fetch.commit() can clear the
    // in-flight bit (write into tracker.next_) at the same moment it
    // applies a mispredict-redirect. The clear is deferred relative to
    // opcoll.resolve_branch() exactly because the scheduler must keep
    // seeing branch_in_flight==true through the cycle where the redirect
    // is applied — otherwise it could issue a shadow instruction from a
    // not-yet-flushed buffer.
    void set_branch_tracker(BranchShadowTracker* tracker) {
        branch_tracker_ = tracker;
    }

    // Phase 5 test hook: explicit override of the redirect-request signal
    // for unit tests that drive FetchStage in isolation. When valid, takes
    // precedence over opcoll_->current_redirect_request().
    void set_redirect_request_override(bool valid, uint32_t warp_id, uint32_t target_pc) {
        redirect_override_valid_ = valid;
        redirect_override_warp_ = warp_id;
        redirect_override_target_ = target_pc;
        has_redirect_override_ = true;
    }
    void clear_redirect_request_override() {
        has_redirect_override_ = false;
        redirect_override_valid_ = false;
    }

    // Test hook: explicit override of the decode-pending-warp signal for unit
    // tests that drive FetchStage without a real DecodeStage. When set,
    // takes precedence over decode_->pending_warp().
    void set_decode_pending_warp_override(std::optional<uint32_t> warp) {
        decode_pending_warp_override_ = warp;
        has_pending_override_ = true;
    }

    // Test hook: explicit override of decode.ready_to_consume_fetch() for
    // unit tests. When set, takes precedence over decode_->ready_to_consume_fetch().
    void set_decode_ready_override(bool ready) {
        decode_ready_override_ = ready;
        has_ready_override_ = true;
    }

private:
    bool query_decode_ready() const;
    std::optional<uint32_t> query_decode_pending_warp() const;
    // Phase 5: applied from commit() when the upstream REGISTERED redirect
    // signal is valid. Mutates committed state (warp PC, instr_buffer,
    // current_output_) — this is correct because commit() is exactly where
    // committed-state updates belong.
    void apply_redirect(uint32_t warp_id, uint32_t target_pc);

    uint32_t num_warps_;
    WarpState* warps_;
    const InstructionMemory& imem_;
    BranchPredictor& predictor_;
    Stats& stats_;
    const DecodeStage* decode_ = nullptr;
    const OperandCollector* opcoll_ = nullptr;
    BranchShadowTracker* branch_tracker_ = nullptr;

    uint32_t rr_pointer_ = 0;
    std::optional<FetchOutput> current_output_;
    std::optional<FetchOutput> next_output_;

    // Test-only overrides; default state is "ready, no pending warp" so a
    // FetchStage exercised in isolation behaves like the previous default.
    bool has_pending_override_ = false;
    std::optional<uint32_t> decode_pending_warp_override_;
    bool has_ready_override_ = false;
    bool decode_ready_override_ = true;

    // Phase 5 test hook: redirect-request override for unit tests.
    bool has_redirect_override_ = false;
    bool redirect_override_valid_ = false;
    uint32_t redirect_override_warp_ = 0;
    uint32_t redirect_override_target_ = 0;
};

} // namespace gpu_sim
