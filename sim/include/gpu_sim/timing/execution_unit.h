#pragma once

#include "gpu_sim/types.h"
#include "gpu_sim/trace_event.h"
#include "gpu_sim/timing/reg.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>

namespace gpu_sim {

// ---------------------------------------------------------------------------
// Phase 10B.0 — scheduler issue-scoreboard constants.
//
// `ExecUnit` (in types.h) has no COUNT sentinel; the tables below are indexed
// by the enum value of the six real units (ALU..SYSTEM, excluding NONE), so
// the count is fixed here. Enum order is ALU, MULTIPLY, DIVIDE, LDST, TLOOKUP,
// SYSTEM, NONE — note LDST precedes TLOOKUP, so the table initializers below
// must follow that order.
constexpr uint32_t kExecUnitCount = 6;  // ALU..SYSTEM

// Fixed iteration / pipeline latencies. DIVIDE and TLOOKUP are iterative units
// (a single busy flag + countdown — see divide_unit.h / tlookup_unit.h); their
// latencies are the single source of truth here and the units reference these
// constants. MULTIPLY is a fully-pipelined unit whose depth is the SimConfig
// field `multiply_pipeline_stages` (default 3). The default constant below is
// used only for default-table documentation/tests; the live WarpScheduler
// computes multiply writeback offsets and bitmap length from the configured
// runtime depth.
constexpr uint32_t kDivideLatency = 32;
constexpr uint32_t kTlookupLatency = 17;
constexpr uint32_t kMulPipelineStages = 3;

inline constexpr uint32_t exec_unit_index(ExecUnit u) {
    return static_cast<uint32_t>(u);
}

// Per-unit structural-hazard iteration latency. 0 => fully pipelined (no
// structural input gate — issue limited only by the writeback bitmap). >0 =>
// iterative: the unit cannot accept a new op for this many cycles after an
// issue to it. Confirmed against the unit sources at Phase 10B.0: ALUUnit and
// MultiplyUnit accept() unconditionally (one op/cycle, fully pipelined);
// DivideUnit and TLookupUnit carry a single busy flag the next accept() would
// clobber, so they are iterative.
constexpr std::array<uint32_t, kExecUnitCount> kUnitIterationLatency = [] {
    std::array<uint32_t, kExecUnitCount> t{};
    t[exec_unit_index(ExecUnit::ALU)]      = 0;                // fully pipelined
    t[exec_unit_index(ExecUnit::MULTIPLY)] = 0;                // fully pipelined
    t[exec_unit_index(ExecUnit::DIVIDE)]   = kDivideLatency;   // iterative
    t[exec_unit_index(ExecUnit::LDST)]     = 0;                // FIFO accounting
    t[exec_unit_index(ExecUnit::TLOOKUP)]  = kTlookupLatency;  // sequential
    t[exec_unit_index(ExecUnit::SYSTEM)]   = 0;                // no fixed unit
    return t;
}();

// Issue -> writeback distance per unit, in cycles for the default configuration.
// END-OF-10B state: every forward edge on the scheduler->opcoll->unit->arbiter
// path is now REGISTERED. 10B.1 (opcoll->unit), 10B.2 (scheduler->opcoll), and
// 10B.3 (unit->arbiter) each added one cycle of pipeline depth, so every
// fixed-latency entry is the 10B.0 land value plus 3. The live scheduler uses
// compute_issue_to_writeback_offset(unit, configured_mul_depth, is_vdot8);
// this default table is retained for tests that explicitly bind the default.
constexpr std::array<uint32_t, kExecUnitCount> kIssueToWritebackOffset = [] {
    std::array<uint32_t, kExecUnitCount> t{};
    t[exec_unit_index(ExecUnit::ALU)]      = 3;                      // 0 + 3 REGISTERED edges
    t[exec_unit_index(ExecUnit::MULTIPLY)] = kMulPipelineStages + 2; // (stages-1) + 3
    t[exec_unit_index(ExecUnit::DIVIDE)]   = kDivideLatency + 2;     // (LATENCY-1) + 3
    t[exec_unit_index(ExecUnit::LDST)]     = 0;                      // never reserves
    t[exec_unit_index(ExecUnit::TLOOKUP)]  = kTlookupLatency + 2;    // (LATENCY-1) + 3
    t[exec_unit_index(ExecUnit::SYSTEM)]   = 0;                      // no writeback
    return t;
}();

inline constexpr uint32_t compute_issue_to_writeback_offset(
    ExecUnit unit, uint32_t multiply_pipeline_stages, bool is_vdot8) {
    uint32_t offset = 0;
    switch (unit) {
        case ExecUnit::ALU:
            offset = 3;
            break;
        case ExecUnit::MULTIPLY:
            offset = multiply_pipeline_stages + 2;
            break;
        case ExecUnit::DIVIDE:
            offset = kDivideLatency + 2;
            break;
        case ExecUnit::TLOOKUP:
            offset = kTlookupLatency + 2;
            break;
        case ExecUnit::LDST:
        case ExecUnit::SYSTEM:
        case ExecUnit::NONE:
        default:
            offset = 0;
            break;
    }
    if (unit == ExecUnit::MULTIPLY && is_vdot8) {
        offset += 1;
    }
    return offset;
}

// Per-issue issue->writeback offset for the default multiply depth. VDOT8
// targets MULTIPLY but spends 2 cycles in the operand collector instead of 1,
// so its writeback lands one cycle later than a plain MUL.
inline constexpr uint32_t compute_issue_to_writeback_offset(ExecUnit unit,
                                                            bool is_vdot8) {
    return compute_issue_to_writeback_offset(unit, kMulPipelineStages, is_vdot8);
}

inline constexpr uint32_t compute_writeback_bitmap_len(uint32_t multiply_pipeline_stages) {
    return 1 + std::max({
        compute_issue_to_writeback_offset(ExecUnit::ALU, multiply_pipeline_stages, false),
        compute_issue_to_writeback_offset(ExecUnit::MULTIPLY, multiply_pipeline_stages, true),
        compute_issue_to_writeback_offset(ExecUnit::DIVIDE, multiply_pipeline_stages, false),
        compute_issue_to_writeback_offset(ExecUnit::TLOOKUP, multiply_pipeline_stages, false),
    });
}

// Default bitmap length. WarpScheduler sizes its live bitmap with the runtime
// multiply depth; this constant remains for default-config test expectations.
constexpr uint32_t kWritebackBitmapLen =
    compute_writeback_bitmap_len(kMulPipelineStages);

struct WritebackEntry {
    bool valid = false;
    uint32_t warp_id = 0;
    uint8_t dest_reg = 0;
    std::array<uint32_t, WARP_SIZE> values{};
    ExecUnit source_unit = ExecUnit::NONE;
    uint32_t pc = 0;
    uint32_t raw_instruction = 0;
    uint64_t issue_cycle = 0;  // For latency tracking
};

// COMBINATIONAL-backward branch-redirect signal (Phase 10E). The producer
// (ALUUnit) asserts next_redirect_ when a branch resolves with misprediction
// during its evaluate(); FetchStage and DecodeStage read next_redirect() at
// the top of their own evaluate() the SAME cycle and flush as needed. Under
// Principle 6 a redirect is a backward control signal: the back-to-front
// evaluate sweep (Phase 10D) runs the ALU before the frontend, so the
// transient is delivered with zero handoff latency and there is no current_*
// twin and no commit() flip. (Phases 10A-10D kept it REGISTERED as an interim
// staging step; 10E converted it to the discipline-correct form.)
struct RedirectRequest {
    bool valid = false;
    uint32_t warp_id = 0;
    uint32_t target_pc = 0;

    // Combinational gate used by every same-cycle consumer of a register
    // whose committed value may belong to a doomed warp this cycle (fetch's
    // READY/STALL gate, fetch's eligibility-scan inflight_to_w term, decode's
    // pull of fetch.current_output, ...). The synthesis-faithful "consumer
    // ANDs its read of Q with !redirect_for_this_warp" mask. Single helper so
    // future consumers can't drift from each other.
    bool targets(uint32_t w) const { return valid && warp_id == w; }
};

class ExecutionUnit {
public:
    virtual ~ExecutionUnit() = default;
    // Back-pressure discipline (REGISTERED + back-pressure direction):
    // current_busy() is a const accessor that reads only the unit's own
    // committed (current_*) state and returns true when the unit cannot
    // accept more work this cycle.
    //
    // Phase 10B.0: the WarpScheduler no longer polls current_busy() for issue
    // gating — it predicts unit availability scheduler-side (the unit_busy_
    // countdowns + the writeback bitmap). current_busy() is retained for the
    // panic-drain query (TimingModel::execution_units_drained) and unit tests.
    // ExecutionUnit is a separate hierarchy from
    // PipelineStage (units produce results consumed by WritebackArbiter
    // rather than participating in the unified evaluate/commit fan-in),
    // but both hierarchies share the discipline. See
    // resources/timing_discipline.md.
    virtual bool current_busy() const = 0;
    // Phase 10B.0.5: explicit double-buffering. seed_next() copies every
    // internal carry-forward field current_* -> next_* at the top of the
    // tick, before evaluate(). It makes the "next_* == current_* on entry to
    // evaluate()" precondition explicit and unconditional, so a stalled cycle
    // (skipped commit()) re-establishes it and the next evaluate() re-runs
    // identically — the prerequisite for the writeback stall (10B.3). Units
    // with no carry-forward state (ALUUnit — 1-cycle latency) implement it as
    // an empty body. Called for every unit at the top of TimingModel::tick(),
    // alongside Scoreboard::seed_next() / BranchShadowTracker::seed_next().
    virtual void seed_next() = 0;
    virtual void evaluate() = 0;
    virtual void commit() = 0;
    virtual void reset() = 0;
    // Phase 10B.3: REGISTERED unit->arbiter edge. current_has_result() reads
    // the unit's committed result-buffer state (current_result_buffer_.valid);
    // the writeback arbiter reads it one cycle after the unit produced the
    // result. consume_result() is a PURE READ — it returns the committed
    // WritebackEntry and mutates nothing. A preempted (unconsumed) result is
    // held across a writeback stall by the stalled unit's gated commit(); the
    // unit's evaluate() overwrites next_result_buffer_ fresh each non-stalled
    // cycle, so a consumed result naturally clears with no invalidation.
    virtual bool current_has_result() const = 0;
    virtual WritebackEntry consume_result() = 0;
    virtual ExecUnit get_type() const = 0;
};

// Test-only ExecutionUnit subclass: a queue of pre-built WritebackEntry values
// that tests inject directly via enqueue() and the WritebackArbiter pops via
// consume_result(). Not used in production — production wb-arbiter sources
// are the five execution units and the LoadGatherBufferFile. Even so, the
// class participates in production wiring through the same WritebackArbiter
// interface as the real units, so discipline-compliance matters even for the
// test source.
//
// Phase 1 (reg-family closeout): the storage is now Reg<std::deque<...>>.
// enqueue() and consume_result() each stage their mutation on next_mut() and
// immediately commit() — the canonical test-fixture pattern (the same shape
// as Reg::initialize() for whole-value writes). This preserves the prior
// "immediate visibility to the arbiter the same cycle the test calls
// enqueue() / consume_result()" contract byte-identically while routing the
// state through a lint-recognized Reg primitive. The WritebackArbiter sees
// queue_.current() through current_has_result(), which is correct. The
// class derives RegisteredStage so the lint's state-shape rule passes;
// reset() delegates to reset_all(), and the seed_next() / commit() overrides
// remain empty bodies because the queue's mutations self-commit inline
// (test-fixture pattern documented at the queue_ declaration below).
class QueuedWritebackSource : public ExecutionUnit, public RegisteredStage {
public:
    explicit QueuedWritebackSource(ExecUnit type) : type_(type) {
        register_state(&queue_);
    }

    void enqueue(const WritebackEntry& entry) {
        if (entry.valid) {
            // Phase 1: stage the push on next_, then commit immediately so
            // the arbiter's same-cycle current_has_result() observes it.
            // This is the test-fixture analog of Reg::initialize(value).
            queue_.next_mut().push_back(entry);
            queue_.commit();
        }
    }

    // No carry-forward double-buffered state to seed beyond the Reg's own
    // auto-seed — same shape as ALUUnit which also implements this as an
    // empty body.
    void seed_next() override {}
    void evaluate() override {}
    void commit() override {}
    void reset() override { reset_all(); }
    // QueuedWritebackSource has no dispatch input; it is never "busy" from
    // the scheduler's perspective (the scheduler never targets it).
    bool current_busy() const override { return false; }
    bool current_has_result() const override { return !queue_.current().empty(); }

    // consume_result() pops the queue front. QueuedWritebackSource is a
    // variable-latency source (not a fixed-latency unit on the writeback
    // bitmap) and is not subject to the writeback stall, so the pop here is
    // not a stalled-cycle re-evaluation hazard.
    //
    // Phase 1 (reg-family closeout): read the front from current(), stage
    // the pop on next_, then commit immediately. The interposed commit is
    // the same test-fixture pattern enqueue() uses — tests calling
    // arbiter.evaluate() (which calls consume_result() inline) expect the
    // pop to be visible the same cycle, with no separately-invoked commit.
    WritebackEntry consume_result() override {
        WritebackEntry entry = queue_.current().front();
        queue_.next_mut().pop_front();
        queue_.commit();
        return entry;
    }

    ExecUnit get_type() const override { return type_; }

    size_t queue_depth() const { return queue_.current().size(); }

private:
    ExecUnit type_;
    // Phase 1 (reg-family closeout): storage is now a Reg<std::deque<...>>.
    // Enrolled in the RegisteredStage mixin so reset_all() drives it; the
    // production-side commit_all() is not called here because the test-only
    // mutators (enqueue / consume_result) self-commit the Reg inline, the
    // shape documented in reg.h's commit() comment as "test-hook dual-write
    // -> tests call seed_next() before arming the staged slot" (this class
    // is one step further: it commits inline so the staged slot becomes
    // current the same instant the test writes it).
    Reg<std::deque<WritebackEntry>> queue_;
};

} // namespace gpu_sim
