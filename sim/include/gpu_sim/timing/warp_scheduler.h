#pragma once

#include "gpu_sim/timing/pipeline_stage.h"
#include "gpu_sim/timing/warp_state.h"
#include "gpu_sim/timing/scoreboard.h"
#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/functional/functional_model.h"
#include "gpu_sim/stats.h"
#include <optional>
#include <functional>

namespace gpu_sim {

struct IssueOutput {
    DecodedInstruction decoded;
    TraceEvent trace;
    uint32_t warp_id;
    uint32_t pc;
};

class WarpScheduler : public PipelineStage {
public:
    // unit_ready: callback to check if a dispatch controller can accept work
    using UnitReadyFn = std::function<bool(ExecUnit)>;

    WarpScheduler(uint32_t num_warps, WarpState* warps, Scoreboard& scoreboard,
                  FunctionalModel& func_model, Stats& stats);

    void evaluate() override;
    void commit() override;
    void reset() override;

    // Set the check for operand collector availability
    void set_opcoll_free(bool free) { opcoll_free_ = free; }

    // Set the callback for execution unit readiness
    void set_unit_ready_fn(UnitReadyFn fn) { unit_ready_fn_ = std::move(fn); }

    std::optional<IssueOutput>& output() { return next_output_; }
    const std::optional<IssueOutput>& current_output() const { return current_output_; }

private:
    bool is_scoreboard_clear(WarpId warp, const DecodedInstruction& d) const;

    uint32_t num_warps_;
    WarpState* warps_;
    Scoreboard& scoreboard_;
    FunctionalModel& func_model_;
    Stats& stats_;

    uint32_t rr_pointer_ = 0;
    bool opcoll_free_ = true;
    UnitReadyFn unit_ready_fn_;

    std::optional<IssueOutput> current_output_;
    std::optional<IssueOutput> next_output_;
};

} // namespace gpu_sim
