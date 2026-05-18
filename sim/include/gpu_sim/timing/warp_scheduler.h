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

// Forward decls. The scheduler retains a non-owning pointer to LdStUnit so it
// can read the REGISTERED FIFO-occupancy accessors (current_fifo_size /
// current_fifo_total_pushes) for the LDST issue gate. Phase 10B.0 removed the
// opcoll busy-poll pointer: unit availability is now predicted scheduler-side
// from issue history (unit_busy_ countdowns + the writeback bitmap).
//
// The ALU / MUL / DIV / TLOOKUP pointers are retained for ONE narrow purpose
// only — the Phase 10B.0 interim writeback-result-buffer issue gate (a
// human-approved deviation, removed in 10B.3): the scheduler reads each unit's
// current_result_pending() to avoid issuing into a unit whose committed
// result buffer is still occupied by an unconsumed result. This is NOT the
// removed busy-poll: it is the narrow current_result_buffer_.valid portion of
// the old current_busy(), retained because 10B.0's bitmap+arbiter cannot yet
// hold a load-preempted fixed-latency result. See current_result_pending()'s
// rationale on each unit and the issue-gate comment in warp_scheduler.cpp.
class LdStUnit;
class ALUUnit;
class MultiplyUnit;
class DivideUnit;
class TLookupUnit;

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

    // Phase 10B.0 wiring: TimingModel calls this once at construction with the
    // scoreboard, branch tracker, the LdSt unit, and the four fixed-latency
    // execution units. The scoreboard and branch tracker drive issue hazards;
    // the LdSt unit is read only through its REGISTERED FIFO-occupancy
    // accessors for the LDST issue gate. The ALU / MUL / DIV / TLOOKUP units
    // are read only through current_result_pending() for the interim
    // writeback-result-buffer issue gate (human-approved deviation, removed in
    // 10B.3 — see the forward-decl comment). All other unit-availability
    // information is now scheduler-side bookkeeping (unit_busy_ /
    // writeback_bitmap_), so the opcoll busy-poll pointer was removed.
    // Pointers may be null in unit tests; null scoreboard/tracker => no
    // hazard, null ldst => the LDST FIFO gate treats the FIFO as empty, null
    // fixed-latency unit => the result-buffer gate treats it as not pending.
    void set_dependencies(Scoreboard* scoreboard,
                          BranchShadowTracker* branch_tracker,
                          LdStUnit* ldst, ALUUnit* alu = nullptr,
                          MultiplyUnit* mul = nullptr, DivideUnit* div = nullptr,
                          TLookupUnit* tlookup = nullptr);

    // Test hooks (Phase 10B.0). Unit tests exercise WarpScheduler without a
    // real opcoll / execution-unit set, so they drive the issue scoreboard
    // directly:
    //  - test_set_unit_busy arms a unit's structural-hazard countdown; the
    //    issue gate then blocks issue to that unit until it drains.
    //  - test_reserve_writeback_slot claims a writeback-bitmap entry at
    //    (bitmap_head_ + offset), so the gate refuses a fixed-latency op whose
    //    predicted writeback cycle collides with the reservation.
    void test_set_unit_busy(ExecUnit unit, uint32_t cycles) {
        unit_busy_[exec_unit_index(unit)] = cycles;
    }
    void test_reserve_writeback_slot(ExecUnit unit, uint32_t offset) {
        writeback_bitmap_[(bitmap_head_ + offset) % kWritebackBitmapLen] = unit;
    }

    std::optional<IssueOutput>& output() { return next_output_; }
    const std::optional<IssueOutput>& current_output() const { return current_output_; }
    const std::array<SchedulerIssueOutcome, MAX_WARPS>& current_diagnostics() const {
        return current_diagnostics_;
    }

private:
    bool is_scoreboard_clear(WarpId warp, const DecodedInstruction& d) const;
    static SchedulerIssueOutcome unit_busy_outcome(ExecUnit unit);
    // Phase 10B.0 interim writeback-result-buffer gate (human-approved
    // deviation, removed in 10B.3). True when the target fixed-latency unit's
    // committed result buffer still holds an unconsumed result; see the
    // definition in warp_scheduler.cpp and current_result_pending() on each
    // unit. Reads the wired alu_ / mul_ / div_ / tlookup_ pointers.
    bool target_result_buffer_occupied(ExecUnit target) const;

    uint32_t num_warps_;
    WarpState* warps_;
    FunctionalModel& func_model_;
    Stats& stats_;

    uint32_t rr_pointer_ = 0;

    // Wired by TimingModel via set_dependencies(). Tests that only construct a
    // WarpScheduler may leave these null (no scoreboard/branch hazard; the
    // LDST FIFO gate treats an absent ldst_ as an empty FIFO; the four
    // fixed-latency unit pointers absent => the interim result-buffer gate
    // treats them as not pending).
    Scoreboard* scoreboard_ = nullptr;
    BranchShadowTracker* branch_tracker_ = nullptr;
    LdStUnit* ldst_ = nullptr;
    // Phase 10B.0 interim writeback-result-buffer gate only (human-approved
    // deviation, removed in 10B.3). Read exclusively via current_result_
    // pending(); NOT the removed busy-poll. See the forward-decl comment.
    ALUUnit* alu_ = nullptr;
    MultiplyUnit* mul_ = nullptr;
    DivideUnit* div_ = nullptr;
    TLookupUnit* tlookup_ = nullptr;

    // ---- Phase 10B.0 issue scoreboard --------------------------------------
    // Per-unit structural-hazard countdown. Set to the unit's iteration
    // latency when an op is issued to it; decremented once per non-frozen
    // cycle at the top of evaluate(); issue to that unit is blocked while > 0.
    // DIVIDE and TLOOKUP use the compile-time kUnitIterationLatency table
    // (fully-pipelined ALU / MULTIPLY have latency 0 and are never armed).
    // LDST also uses this countdown: although LDST has no fixed compile-time
    // latency (its addr-gen latency is the runtime ldst_iteration_latency_
    // below), its address-generation stage holds exactly one op at a time —
    // LdStUnit::accept() unconditionally overwrites next_pending_entry_, so
    // issuing a second LDST op before the first leaves addr-gen would clobber
    // an in-flight op. The countdown spaces consecutive LDST issues by the
    // addr-gen latency to prevent that. (Pre-10B.0 this was enforced by the
    // now-removed ldst->current_busy() poll; the FIFO-occupancy gate alone
    // models FIFO capacity, not the single addr-gen slot.)
    std::array<uint32_t, kExecUnitCount> unit_busy_{};

    // LDST address-generation iteration latency, ceil(WARP_SIZE /
    // num_ldst_units). Captured from the wired LdSt unit in set_dependencies()
    // because it is a runtime-configured value (unlike DIVIDE / TLOOKUP whose
    // latencies are compile-time constants in kUnitIterationLatency). 0 when no
    // LdSt unit is wired (unit-test default) — LDST issue then has no addr-gen
    // structural gate, matching the empty-FIFO default.
    uint32_t ldst_iteration_latency_ = 0;

    // Writeback-slot bitmap — the binding fixed-latency writeback schedule. A
    // circular array; each occupied entry marks a near-future cycle on which a
    // fixed-latency writeback is already promised. An issue reserves the entry
    // at (bitmap_head_ + offset); the gate refuses any fixed-latency op whose
    // predicted writeback cycle is already claimed. bitmap_head_ advances one
    // slot per non-frozen cycle.
    std::array<std::optional<ExecUnit>, kWritebackBitmapLen> writeback_bitmap_{};
    uint32_t bitmap_head_ = 0;

    // LDST FIFO-occupancy accounting: a scheduler-side monotonic count of ops
    // ever issued to LDST. The in-transit population is the difference
    // (ldst_issued_total_ - ldst_->current_fifo_total_pushes()).
    uint32_t ldst_issued_total_ = 0;

    // Interim operand-collector cooldown. The opcoll takes 2 cycles for VDOT8
    // and 1 cycle for everything else; set on issue and decremented at the top
    // of evaluate(). A cooldown of 1 (every non-VDOT8 op) clears by the next
    // cycle and permits a fresh issue every cycle; a cooldown of 2 (VDOT8)
    // blocks exactly one cycle. Drops out when opcoll becomes always-1-cycle.
    uint32_t opcoll_cooldown_cycles_ = 0;

    std::optional<IssueOutput> current_output_;
    std::optional<IssueOutput> next_output_;
    std::array<SchedulerIssueOutcome, MAX_WARPS> current_diagnostics_{};
    std::array<SchedulerIssueOutcome, MAX_WARPS> next_diagnostics_{};
};

} // namespace gpu_sim
