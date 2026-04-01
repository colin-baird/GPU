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

    bool is_free() const { return !busy_; }

    // Accept new instruction from scheduler
    void accept(const IssueOutput& issue);

    // Output to dispatch
    std::optional<DispatchInput>& output() { return next_output_; }
    const std::optional<DispatchInput>& current_output() const { return current_output_; }
    bool busy() const { return busy_; }
    uint32_t cycles_remaining() const { return cycles_remaining_; }
    std::optional<uint32_t> resident_warp() const {
        if (!busy_) return std::nullopt;
        return current_instr_.warp_id;
    }
    const IssueOutput* current_instruction() const {
        return busy_ ? &current_instr_ : nullptr;
    }

private:
    Stats& stats_;
    bool busy_ = false;
    uint32_t cycles_remaining_ = 0;
    IssueOutput current_instr_;

    std::optional<DispatchInput> current_output_;
    std::optional<DispatchInput> next_output_;
};

} // namespace gpu_sim
