#pragma once

#include "gpu_sim/types.h"
#include "gpu_sim/trace_event.h"
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
// field `multiply_pipeline_stages` (default 3). The compile-time
// kMulPipelineStages constant is the canonical depth the scheduler's writeback
// bitmap is sized against; the runtime MultiplyUnit still honors its config
// field. A config that overrode multiply_pipeline_stages above this value
// would need a wider bitmap — not exercised by any current config/test.
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

// Issue -> writeback distance per unit, in cycles. END-OF-10B state: every
// forward edge on the scheduler->opcoll->unit->arbiter path is now REGISTERED.
// 10B.1 (opcoll->unit), 10B.2 (scheduler->opcoll), and 10B.3 (unit->arbiter)
// each added one cycle of pipeline depth, so every fixed-latency entry is the
// 10B.0 land value plus 3. The arbiter-side count_fixed_with_result() <= 1
// assert (writeback_arbiter.cpp) is the live check that these offsets are
// exact — a trip means an entry is wrong.
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

// Length of the circular writeback-slot bitmap. Strictly greater than the
// largest issue->writeback offset so a max-offset reservation never aliases
// bitmap_head_ itself (the slot cleared this cycle). VDOT8 spends an extra
// opcoll cycle, so MULTIPLY's effective offset is one larger — accounted here.
constexpr uint32_t kWritebackBitmapLen =
    1 + std::max({kIssueToWritebackOffset[exec_unit_index(ExecUnit::ALU)],
                  kIssueToWritebackOffset[exec_unit_index(ExecUnit::MULTIPLY)] + 1,
                  kIssueToWritebackOffset[exec_unit_index(ExecUnit::DIVIDE)],
                  kIssueToWritebackOffset[exec_unit_index(ExecUnit::TLOOKUP)]});

// Per-issue issue->writeback offset. VDOT8 targets MULTIPLY but spends 2 cycles
// in the operand collector instead of 1, so its writeback lands one cycle
// later than a plain MUL — add 1 to MULTIPLY's table value. is_vdot8 is
// derived at the issue site from the decoded instruction (num_src_regs == 3).
inline constexpr uint32_t compute_issue_to_writeback_offset(ExecUnit unit,
                                                            bool is_vdot8) {
    uint32_t offset = kIssueToWritebackOffset[exec_unit_index(unit)];
    if (unit == ExecUnit::MULTIPLY && is_vdot8) {
        offset += 1;
    }
    return offset;
}

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

class QueuedWritebackSource : public ExecutionUnit {
public:
    explicit QueuedWritebackSource(ExecUnit type) : type_(type) {}

    void enqueue(const WritebackEntry& entry) {
        if (entry.valid) {
            queue_.push_back(entry);
        }
    }

    // No carry-forward double-buffered state — the queue is mutated directly
    // by enqueue()/consume_result(), not via a next_* slot — so seed_next()
    // is empty (same category as ALUUnit).
    void seed_next() override {}
    void evaluate() override {}
    void commit() override {}
    void reset() override { queue_.clear(); }
    // QueuedWritebackSource has no dispatch input; it is never "busy" from
    // the scheduler's perspective (the scheduler never targets it).
    bool current_busy() const override { return false; }
    bool current_has_result() const override { return !queue_.empty(); }

    // consume_result() pops the queue front. QueuedWritebackSource is a
    // variable-latency source (not a fixed-latency unit on the writeback
    // bitmap) and is not subject to the writeback stall, so the pop here is
    // not a stalled-cycle re-evaluation hazard.
    WritebackEntry consume_result() override {
        WritebackEntry entry = queue_.front();
        queue_.pop_front();
        return entry;
    }

    ExecUnit get_type() const override { return type_; }

    size_t queue_depth() const { return queue_.size(); }

private:
    ExecUnit type_;
    std::deque<WritebackEntry> queue_;
};

} // namespace gpu_sim
