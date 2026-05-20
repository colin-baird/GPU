#pragma once

#include "gpu_sim/timing/pipeline_stage.h"
#include "gpu_sim/timing/reg.h"
#include "gpu_sim/timing/warp_state.h"
#include "gpu_sim/timing/scoreboard.h"
#include "gpu_sim/timing/branch_shadow_tracker.h"
#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/functional/functional_model.h"
#include "gpu_sim/stats.h"
#include <optional>
#include <array>
#include <vector>

namespace gpu_sim {

// Forward decls. The scheduler retains a non-owning pointer to LdStUnit so it
// can read the REGISTERED FIFO-occupancy accessors (current_fifo_size /
// current_fifo_total_pushes) for the LDST issue gate. Phase 10B.0 removed the
// opcoll busy-poll pointer: unit availability is now predicted scheduler-side
// from issue history (unit_busy_ countdowns + the writeback bitmap).
//
// Phase 10B.3 removed the interim writeback-result-buffer issue gate (and the
// ALU/MUL/DIV/TLOOKUP pointers it needed) — the combinational-backward
// writeback stall now holds a load-preempted fixed-latency result, so the
// scheduler no longer reads any execution unit. It retains a WritebackArbiter
// pointer to read next_writeback_stall() (the stall freeze guard).
class LdStUnit;
class WritebackArbiter;

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

class WarpScheduler : public PipelineStage, public RegisteredStage {
public:
    WarpScheduler(uint32_t num_warps, WarpState* warps,
                  FunctionalModel& func_model, Stats& stats,
                  uint32_t multiply_pipeline_stages = kMulPipelineStages);

    // Phase 4 (reg.h migration): seed all registered state at the top of the
    // tick. TimingModel::tick() now calls scheduler_->seed_next() alongside the
    // other stages. Pre-Phase-4 the scheduler had no seed_next() and relied on
    // the post-commit equality (next == current) carrying its in-place-mutated
    // registers forward; seed_all() re-establishes that equality explicitly and
    // is therefore byte-identical for every register on a non-stalled cycle.
    // On a writeback-stall cycle evaluate() early-returns (no decrements) and
    // commit() is gated (no flip), so seed_all() simply re-runs idempotently.
    void seed_next() { seed_all(); }
    void evaluate() override;
    void commit() override;
    void reset() override;

    // Panic flush hook. Called from TimingModel::tick() at the commit-phase
    // boundary when the panic signal becomes active. Currently delegates to
    // reset(); kept as a separate name so the call site reads as a
    // panic-cascade event rather than a state reset.
    void flush();

    // Phase 10B.0/10B.3 wiring: TimingModel calls this once at construction
    // with the scoreboard, branch tracker, and the LdSt unit. The scoreboard
    // and branch tracker drive issue hazards; the LdSt unit is read only
    // through its REGISTERED FIFO-occupancy accessors for the LDST issue gate.
    // All other unit-availability information is scheduler-side bookkeeping
    // (unit_busy_ / writeback_bitmap_). Phase 10B.3 removed the interim
    // writeback-result-buffer gate, so no execution-unit pointers are wired
    // here any more. Pointers may be null in unit tests; null
    // scoreboard/tracker => no hazard, null ldst => the LDST FIFO gate treats
    // the FIFO as empty.
    void set_dependencies(Scoreboard* scoreboard,
                          BranchShadowTracker* branch_tracker,
                          LdStUnit* ldst);

    // Phase 10B.3: wire the WritebackArbiter so evaluate() can read
    // next_writeback_stall() (the early-return freeze guard) and commit() can
    // gate the next_->current_ flip. Null in unit tests => no stall ever.
    void set_writeback_arbiter(WritebackArbiter* arbiter) {
        wb_arbiter_ = arbiter;
    }

    // Test hooks (Phase 10B.0). Unit tests exercise WarpScheduler without a
    // real opcoll / execution-unit set, so they drive the issue scoreboard
    // directly:
    //  - test_set_unit_busy arms a unit's structural-hazard countdown; the
    //    issue gate then blocks issue to that unit until it drains.
    //  - test_reserve_writeback_slot claims a writeback-bitmap entry at
    //    (bitmap_head_ + offset), so the gate refuses a fixed-latency op whose
    //    predicted writeback cycle collides with the reservation.
    // The hooks write to the staged slot only. The required setup order is:
    //   1) f.scheduler->seed_next();       // copies committed -> staged
    //   2) f.scheduler->test_set_unit_busy(...);   // overwrites staged
    //   3) f.scheduler->evaluate();        // mutates staged in place
    // Calling seed_next() AFTER a hook would clobber the armed value (seed
    // copies committed -> staged). Pre-Phase-1-of-current_mut()-removal the
    // hooks dual-wrote committed AND staged to be order-insensitive; the
    // committed-side write was a Q-mutation that the current_mut() anti-pattern
    // refactor eliminates.
    void test_set_unit_busy(ExecUnit unit, uint32_t cycles) {
        unit_busy_.next_mut()[exec_unit_index(unit)] = cycles;
    }
    void test_reserve_writeback_slot(ExecUnit unit, uint32_t offset) {
        writeback_bitmap_.next_mut()[bitmap_slot(offset)] = unit;
    }

    // Phase 10B.2: REGISTERED scheduler->opcoll output. current_output()
    // returns the committed slot; the OperandCollector pulls it at the top of
    // its evaluate(). evaluate() continues to write next_output_.
    const std::optional<IssueOutput>& current_output() const { return output_.current(); }
    const std::array<SchedulerIssueOutcome, MAX_WARPS>& current_diagnostics() const {
        return diagnostics_.current();
    }

private:
    bool is_scoreboard_clear(WarpId warp, const DecodedInstruction& d) const;
    static SchedulerIssueOutcome unit_busy_outcome(ExecUnit unit);
    uint32_t issue_to_writeback_offset(ExecUnit unit, bool is_vdot8) const;
    size_t bitmap_slot(uint32_t offset) const;

    uint32_t num_warps_;                 // config
    WarpState* warps_;                   // config (back-pointer)
    FunctionalModel& func_model_;        // config (back-pointer)
    Stats& stats_;                       // config (back-pointer)
    uint32_t multiply_pipeline_stages_;  // config

    // Wired by TimingModel via set_dependencies(). Tests that only construct a
    // WarpScheduler may leave these null (no scoreboard/branch hazard; the
    // LDST FIFO gate treats an absent ldst_ as an empty FIFO).
    Scoreboard* scoreboard_ = nullptr;
    BranchShadowTracker* branch_tracker_ = nullptr;
    LdStUnit* ldst_ = nullptr;
    // Phase 10B.3: wired via set_writeback_arbiter(). Null in unit tests => no
    // writeback stall ever (evaluate() never early-returns, commit() never
    // gates).
    WritebackArbiter* wb_arbiter_ = nullptr;

    // ---- Phase 10B.0 issue scoreboard --------------------------------------
    // Phase 4 (reg.h migration): every cross-cycle register is a Reg<T>. The
    // in-place-mutated countdowns and bookkeeping below were single-buffered
    // pre-Phase-4; their faithful Reg translation is `next_mut()` to apply the
    // top-of-evaluate mutation (decrement, advance, reserve) and `next()` to
    // read back the post-mutation value during the same evaluate() — which is
    // what every issue gate already does (e.g. `if (unit_busy_[i] > 0)` is
    // read AFTER the top-of-evaluate `unit_busy_[i]--`). seed_all() at the
    // top of the tick copies current -> next so the in-place mutation is
    // applied to a freshly-seeded value, then commit_all() at end-of-cycle
    // latches the result; on a writeback-stall cycle the early-returned
    // evaluate() does not mutate next and the gated commit_all() does not
    // flip, so the cycle is genuinely frozen.

    // Round-robin warp pointer. Advances one slot per non-frozen cycle at the
    // end of evaluate(); the issue scan begins at (rr_pointer + i) % num_warps.
    Reg<uint32_t> rr_pointer_;

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
    Reg<std::array<uint32_t, kExecUnitCount>> unit_busy_;

    // LDST address-generation iteration latency, ceil(WARP_SIZE /
    // num_ldst_units). Captured from the wired LdSt unit in set_dependencies()
    // because it is a runtime-configured value (unlike DIVIDE / TLOOKUP whose
    // latencies are compile-time constants in kUnitIterationLatency). 0 when no
    // LdSt unit is wired (unit-test default) — LDST issue then has no addr-gen
    // structural gate, matching the empty-FIFO default.
    uint32_t ldst_iteration_latency_ = 0;  // config

    // Writeback-slot bitmap — the binding fixed-latency writeback schedule. A
    // circular array sized from the configured fixed-latency offsets; each
    // occupied entry marks a near-future cycle on which a fixed-latency
    // writeback is already promised. An issue reserves the entry at
    // (bitmap_head_ + offset); the gate refuses any fixed-latency op whose
    // predicted writeback cycle is already claimed. bitmap_head_ advances one
    // slot per non-frozen cycle.
    Reg<std::vector<std::optional<ExecUnit>>> writeback_bitmap_;
    Reg<uint32_t> bitmap_head_;

    // LDST FIFO-occupancy accounting: a scheduler-side monotonic count of ops
    // ever issued to LDST. The in-transit population is the difference
    // (ldst_issued_total_ - ldst_->current_fifo_total_pushes()). Plain
    // uint32_t: monotonic accumulator, not a clocked hardware register.
    uint32_t ldst_issued_total_ = 0;  // sim-instrumentation

    // Interim operand-collector cooldown. The opcoll takes 2 cycles for VDOT8
    // and 1 cycle for everything else; set on issue and decremented at the top
    // of evaluate(). A cooldown of 1 (every non-VDOT8 op) clears by the next
    // cycle and permits a fresh issue every cycle; a cooldown of 2 (VDOT8)
    // blocks exactly one cycle. Drops out when opcoll becomes always-1-cycle.
    Reg<uint32_t> opcoll_cooldown_cycles_;

    Reg<std::optional<IssueOutput>> output_;
    Reg<std::array<SchedulerIssueOutcome, MAX_WARPS>> diagnostics_;
};

} // namespace gpu_sim
