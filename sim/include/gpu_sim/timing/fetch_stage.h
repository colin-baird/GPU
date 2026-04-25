#pragma once

#include "gpu_sim/timing/pipeline_stage.h"
#include "gpu_sim/timing/branch_predictor.h"
#include "gpu_sim/timing/warp_state.h"
#include "gpu_sim/functional/memory.h"
#include "gpu_sim/stats.h"
#include <optional>

namespace gpu_sim {

class DecodeStage;  // forward decl: fetch reads decode.ready_to_consume_fetch()

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

    // Branch redirect: called by execute stage
    void redirect_warp(uint32_t warp_id, uint32_t target_pc);

    // Wire decode after both stages are constructed. Fetch reads
    // decode->ready_to_consume_fetch() and decode->pending_warp() during
    // evaluate() as READY/STALL signals (Phase 3 discipline).
    void set_decode(const DecodeStage* decode) { decode_ = decode; }

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

    uint32_t num_warps_;
    WarpState* warps_;
    const InstructionMemory& imem_;
    BranchPredictor& predictor_;
    Stats& stats_;
    const DecodeStage* decode_ = nullptr;

    uint32_t rr_pointer_ = 0;
    std::optional<FetchOutput> current_output_;
    std::optional<FetchOutput> next_output_;

    // Test-only overrides; default state is "ready, no pending warp" so a
    // FetchStage exercised in isolation behaves like the previous default.
    bool has_pending_override_ = false;
    std::optional<uint32_t> decode_pending_warp_override_;
    bool has_ready_override_ = false;
    bool decode_ready_override_ = true;
};

} // namespace gpu_sim
