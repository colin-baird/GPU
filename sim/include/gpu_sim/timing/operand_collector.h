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

private:
    Stats& stats_;
    bool busy_ = false;
    uint32_t cycles_remaining_ = 0;
    IssueOutput current_instr_;

    std::optional<DispatchInput> current_output_;
    std::optional<DispatchInput> next_output_;
};

} // namespace gpu_sim
