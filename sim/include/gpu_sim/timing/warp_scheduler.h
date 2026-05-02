#pragma once

#include "gpu_sim/timing/pipeline_stage.h"
#include "gpu_sim/timing/warp_state.h"
#include "gpu_sim/timing/scoreboard.h"
#include "gpu_sim/timing/branch_shadow_tracker.h"
#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/functional/functional_model.h"
#include "gpu_sim/stats.h"
#include <optional>
#include <array>

namespace gpu_sim {

// Forward decls: scheduler holds non-owning pointers to the opcoll and the
// five typed execution units so it can read each one's current_busy() directly,
// replacing the Phase-3-era set_opcoll_free / set_unit_ready_fn setter pair.
class OperandCollector;
class ALUUnit;
class MultiplyUnit;
class DivideUnit;
class TLookupUnit;
class LdStUnit;

struct IssueOutput {
    DecodedInstruction decoded;
    TraceEvent trace;
    uint32_t warp_id;
    uint32_t pc;
    BranchPrediction prediction;
};

enum class SchedulerIssueOutcome {
    INACTIVE,
    BUFFER_EMPTY,
    BRANCH_SHADOW,
    SCOREBOARD,
    OPCOLL_BUSY,
    UNIT_BUSY_ALU,
    UNIT_BUSY_MULTIPLY,
    UNIT_BUSY_DIVIDE,
    UNIT_BUSY_TLOOKUP,
    UNIT_BUSY_LDST,
    READY_NOT_SELECTED,
    ISSUED
};

class WarpScheduler : public PipelineStage {
public:
    WarpScheduler(uint32_t num_warps, WarpState* warps,
                  FunctionalModel& func_model, Stats& stats);

    void evaluate() override;
    void commit() override;
    void reset() override;

    // Panic flush hook. Called from TimingModel::tick() at the commit-phase
    // boundary when the panic signal becomes active. Currently delegates to
    // reset(); kept as a separate name so the call site reads as a
    // panic-cascade event rather than a state reset.
    void flush();

    // Phase 2 wiring: TimingModel calls this once at construction with the
    // scoreboard, branch tracker, opcoll, and the five typed execution
    // units. Reference-typed dependencies (scoreboard_, branch_tracker_)
    // were promoted to pointers in Phase 2 of the naming-and-access-
    // discipline refactor so every cross-stage receiver field has a
    // uniform raw-pointer shape (post-construction wiring, may be null
    // in unit tests). evaluate() reads each consumer's current_busy()
    // and the scoreboard/tracker via these wired pointers.
    void set_dependencies(Scoreboard* scoreboard,
                          BranchShadowTracker* branch_tracker,
                          OperandCollector* opcoll, ALUUnit* alu,
                          MultiplyUnit* mul, DivideUnit* div,
                          TLookupUnit* tlookup, LdStUnit* ldst) {
        scoreboard_ = scoreboard;
        branch_tracker_ = branch_tracker;
        opcoll_ = opcoll;
        alu_ = alu;
        mul_ = mul;
        div_ = div;
        tlookup_ = tlookup;
        ldst_ = ldst;
    }

    // Test hooks: explicit overrides for unit tests that exercise
    // WarpScheduler without a real opcoll / execution-unit set. When set,
    // they take precedence over the wired consumer's current_busy(). With
    // neither override set and no consumers wired (default-constructed),
    // the scheduler treats opcoll/unit as ready — matching the prior
    // default behavior of test fixtures that never called the setters.
    void set_opcoll_ready_override(std::optional<bool> ready) {
        opcoll_ready_override_ = ready;
    }
    void set_unit_ready_override(ExecUnit unit, std::optional<bool> ready) {
        switch (unit) {
            case ExecUnit::ALU:      alu_ready_override_ = ready; break;
            case ExecUnit::MULTIPLY: mul_ready_override_ = ready; break;
            case ExecUnit::DIVIDE:   div_ready_override_ = ready; break;
            case ExecUnit::TLOOKUP:  tlookup_ready_override_ = ready; break;
            case ExecUnit::LDST:     ldst_ready_override_ = ready; break;
            default: break;
        }
    }

    std::optional<IssueOutput>& output() { return next_output_; }
    const std::optional<IssueOutput>& current_output() const { return current_output_; }
    const std::array<SchedulerIssueOutcome, MAX_WARPS>& current_diagnostics() const {
        return current_diagnostics_;
    }

private:
    bool is_scoreboard_clear(WarpId warp, const DecodedInstruction& d) const;
    static SchedulerIssueOutcome unit_busy_outcome(ExecUnit unit);
    bool query_opcoll_ready() const;
    bool query_unit_ready(ExecUnit unit) const;

    uint32_t num_warps_;
    WarpState* warps_;
    FunctionalModel& func_model_;
    Stats& stats_;

    uint32_t rr_pointer_ = 0;

    // Wired by TimingModel via set_dependencies(). Tests that only
    // construct a WarpScheduler may leave these null and rely on
    // overrides (or the default "all ready" / "no scoreboard hazard"
    // behavior).
    Scoreboard* scoreboard_ = nullptr;
    BranchShadowTracker* branch_tracker_ = nullptr;
    OperandCollector* opcoll_ = nullptr;
    ALUUnit* alu_ = nullptr;
    MultiplyUnit* mul_ = nullptr;
    DivideUnit* div_ = nullptr;
    TLookupUnit* tlookup_ = nullptr;
    LdStUnit* ldst_ = nullptr;

    std::optional<bool> opcoll_ready_override_;
    std::optional<bool> alu_ready_override_;
    std::optional<bool> mul_ready_override_;
    std::optional<bool> div_ready_override_;
    std::optional<bool> tlookup_ready_override_;
    std::optional<bool> ldst_ready_override_;

    std::optional<IssueOutput> current_output_;
    std::optional<IssueOutput> next_output_;
    std::array<SchedulerIssueOutcome, MAX_WARPS> current_diagnostics_{};
    std::array<SchedulerIssueOutcome, MAX_WARPS> next_diagnostics_{};
};

} // namespace gpu_sim
