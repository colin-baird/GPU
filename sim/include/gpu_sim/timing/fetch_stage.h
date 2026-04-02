#pragma once

#include "gpu_sim/timing/pipeline_stage.h"
#include "gpu_sim/timing/branch_predictor.h"
#include "gpu_sim/timing/warp_state.h"
#include "gpu_sim/functional/memory.h"
#include "gpu_sim/stats.h"
#include <optional>

namespace gpu_sim {

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

    // Decode stage signals it consumed the current output
    void consume_output() { output_consumed_ = true; }

private:
    uint32_t num_warps_;
    WarpState* warps_;
    const InstructionMemory& imem_;
    BranchPredictor& predictor_;
    Stats& stats_;

    uint32_t rr_pointer_ = 0;
    bool output_consumed_ = true;
    std::optional<FetchOutput> current_output_;
    std::optional<FetchOutput> next_output_;
};

} // namespace gpu_sim
