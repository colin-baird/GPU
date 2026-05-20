#include "gpu_sim/timing/warp_scheduler.h"

#include "gpu_sim/timing/ldst_unit.h"
#include "gpu_sim/timing/writeback_arbiter.h"

#include <algorithm>
#include <stdexcept>

namespace gpu_sim {

SchedulerIssueOutcome WarpScheduler::unit_busy_outcome(ExecUnit unit) {
    switch (unit) {
        case ExecUnit::ALU:      return SchedulerIssueOutcome::UNIT_BUSY_ALU;
        case ExecUnit::MULTIPLY: return SchedulerIssueOutcome::UNIT_BUSY_MULTIPLY;
        case ExecUnit::DIVIDE:   return SchedulerIssueOutcome::UNIT_BUSY_DIVIDE;
        case ExecUnit::TLOOKUP:  return SchedulerIssueOutcome::UNIT_BUSY_TLOOKUP;
        case ExecUnit::LDST:     return SchedulerIssueOutcome::UNIT_BUSY_LDST;
        default:                 return SchedulerIssueOutcome::OPCOLL_BUSY;
    }
}

WarpScheduler::WarpScheduler(uint32_t num_warps, WarpState* warps,
                             FunctionalModel& func_model, Stats& stats,
                             uint32_t multiply_pipeline_stages)
    : num_warps_(num_warps), warps_(warps),
      func_model_(func_model), stats_(stats),
      multiply_pipeline_stages_(multiply_pipeline_stages) {
    if (multiply_pipeline_stages_ == 0) {
        throw std::invalid_argument("multiply_pipeline_stages must be >= 1");
    }
    // Size BOTH slots of the bitmap register so seed_all() / commit_all() copy
    // same-sized vectors and so the in-place mutation in evaluate() addresses a
    // sized buffer.
    const size_t bitmap_len =
        compute_writeback_bitmap_len(multiply_pipeline_stages_);
    writeback_bitmap_.initialize(std::vector<std::optional<ExecUnit>>(
        bitmap_len, std::nullopt));
    register_state(&rr_pointer_, &unit_busy_, &writeback_bitmap_,
                   &bitmap_head_, &opcoll_cooldown_cycles_, &output_,
                   &diagnostics_);
}

void WarpScheduler::set_dependencies(Scoreboard* scoreboard,
                                     BranchShadowTracker* branch_tracker,
                                     LdStUnit* ldst) {
    scoreboard_ = scoreboard;
    branch_tracker_ = branch_tracker;
    ldst_ = ldst;
    // Capture the LDST addr-gen iteration latency (runtime-configured value;
    // see warp_scheduler.h). Defined here rather than inline in the header so
    // LdStUnit stays an incomplete type at the header.
    ldst_iteration_latency_ = ldst_ ? ldst_->current_addr_gen_latency() : 0;
}

bool WarpScheduler::is_scoreboard_clear(WarpId warp, const DecodedInstruction& d) const {
    // No scoreboard wired (unit-test default): no register hazards.
    if (scoreboard_ == nullptr) return true;
    // RAW: every source operand must have no in-flight write.
    if (d.num_src_regs >= 1 && scoreboard_->current_pending(warp, d.rs1)) return false;
    if (d.num_src_regs >= 2 && scoreboard_->current_pending(warp, d.rs2)) return false;
    // RAW on the rd operand of 3-operand instructions (VDOT8 accumulator).
    if (d.reads_rd && scoreboard_->current_pending(warp, d.rd)) return false;
    // WAW: a write to rd must not race an earlier in-flight write to the same
    // register. Execution-unit latencies are asymmetric (ALU 3, MUL 5, DIV 34,
    // TLOOKUP 19, LDST variable), so a later instruction can reach the
    // writeback arbiter before an earlier one — two pending writes to one rd
    // could otherwise retire out of program order. Stalling issue until the
    // prior write commits keeps per-(warp, register) writeback in program
    // order. (For VDOT8 the rd-as-source check above already covers this.)
    if (d.has_rd && d.rd != 0 && scoreboard_->current_pending(warp, d.rd)) return false;
    return true;
}

uint32_t WarpScheduler::issue_to_writeback_offset(ExecUnit unit, bool is_vdot8) const {
    return compute_issue_to_writeback_offset(unit, multiply_pipeline_stages_, is_vdot8);
}

size_t WarpScheduler::bitmap_slot(uint32_t offset) const {
    // Phase 4 (reg.h migration): evaluate() mutates bitmap_head_ in-place at
    // the top (advance by 1), then calls bitmap_slot() to address the wrap;
    // the post-advance value is what every in-evaluate caller needs, so read
    // the staged slot. Cross-stage callers do not exist (this helper is
    // private). Also called by the test_reserve_writeback_slot hook, where the
    // pre-evaluate bitmap_head_ is still 0 in both slots — same answer either
    // way.
    return (bitmap_head_.next() + offset) % writeback_bitmap_.next().size();
}

void WarpScheduler::evaluate() {
    // Phase 10B.3: writeback-stall freeze guard. The arbiter (sequenced first
    // in the evaluate sweep) asserts next_writeback_stall() when a load
    // preempted a fixed-latency writeback. On a stalled cycle the scheduler
    // early-returns: it issues nothing, advances no issue bookkeeping
    // (bitmap_head_, reservations, unit_busy_, opcoll_cooldown_, the LDST
    // counters), and its commit() is gated — so the writeback schedule freezes
    // in lockstep with the pipeline it models. No instruction is lost: the
    // already-issued instruction is held in current_output_ by the gated
    // commit(). The body below never runs on a stalled cycle, so the
    // scheduler's Stats increments stay safely in evaluate().
    if (wb_arbiter_ != nullptr && wb_arbiter_->next_writeback_stall()) {
        return;
    }

    output_.set_next(std::nullopt);
    diagnostics_.next_mut().fill(SchedulerIssueOutcome::INACTIVE);

    // ---- Top-of-evaluate issue-scoreboard bookkeeping ----------------------
    // Phase 10B.0: the scheduler predicts unit availability from issue history
    // rather than polling its consumers. This bookkeeping runs once per cycle,
    // before the issue gate reads any of it, so a value set on issue at cycle
    // N is checked one lower at N+1 (the decrement-then-check ordering that
    // makes the countdowns exact). Phase 4 (reg.h migration): seed_all() at
    // the top of the tick has copied current_* -> next_*, so the in-place
    // mutations below apply to a freshly-seeded staged value.
    auto& bitmap_next = writeback_bitmap_.next_mut();
    bitmap_next[bitmap_head_.next()] = std::nullopt;  // reserved cycle elapsed
    bitmap_head_.next_mut() = (bitmap_head_.next() + 1) % bitmap_next.size();
    for (auto& b : unit_busy_.next_mut()) {
        if (b > 0) --b;                                // DIVIDE / TLOOKUP only
    }
    if (opcoll_cooldown_cycles_.next() > 0) {
        --opcoll_cooldown_cycles_.next_mut();
    }

    bool issued = false;
    bool any_buffer_empty = false;
    bool any_active = false;
    uint32_t selected_warp = 0;
    BufferEntry selected_entry{};
    bool has_selected_entry = false;

    for (uint32_t i = 0; i < num_warps_; ++i) {
        uint32_t w = (rr_pointer_.next() + i) % num_warps_;

        if (!warps_[w].active) {
            diagnostics_.next_mut()[w] = SchedulerIssueOutcome::INACTIVE;
            continue;
        }
        any_active = true;

        if (warps_[w].instr_buffer.is_empty()) {
            stats_.warp_stall_buffer_empty[w]++;
            diagnostics_.next_mut()[w] = SchedulerIssueOutcome::BUFFER_EMPTY;
            any_buffer_empty = true;
            continue;
        }

        // Phase 5 REGISTERED: read tracker.current_in_flight(w) which exposes
        // current_ (committed) state. The note_branch_issued write below
        // goes into next_; commit() flips at end of cycle. No tracker
        // wired (unit-test default): no branch shadow ever in flight.
        if (branch_tracker_ != nullptr && branch_tracker_->current_in_flight(w)) {
            stats_.warp_stall_branch_shadow[w]++;
            diagnostics_.next_mut()[w] = SchedulerIssueOutcome::BRANCH_SHADOW;
            continue;
        }

        const auto& entry = warps_[w].instr_buffer.front();
        const DecodedInstruction& decoded = entry.decoded;

        if (!is_scoreboard_clear(w, decoded)) {
            stats_.warp_stall_scoreboard[w]++;
            diagnostics_.next_mut()[w] = SchedulerIssueOutcome::SCOREBOARD;
            continue;
        }

        // Phase 10B.0 combined issue gate ------------------------------------
        const ExecUnit target = decoded.target_unit;
        const bool is_vdot8   = (decoded.num_src_regs == 3);
        const bool writes_back = decoded.has_rd && decoded.rd != 0;
        // offset is read by both the bitmap conflict check and the reservation
        // at issue. Harmless for LDST / SYSTEM (offset 0, never reserve).
        const uint32_t offset = issue_to_writeback_offset(target, is_vdot8);

        if (target == ExecUnit::LDST) {
            // (1) Addr-gen structural hazard. LdStUnit's address-generation
            // stage holds one op at a time and accept() unconditionally
            // overwrites it, so consecutive LDST issues must be spaced by the
            // addr-gen latency or the second clobbers an in-flight op. The
            // unit_busy_[LDST] countdown enforces that spacing.
            if (unit_busy_.next()[exec_unit_index(ExecUnit::LDST)] > 0) {
                stats_.warp_stall_unit_busy[w]++;
                stats_.scheduler_unit_busy_stall_cycles[
                    exec_unit_index(ExecUnit::LDST)]++;
                diagnostics_.next_mut()[w] = SchedulerIssueOutcome::UNIT_BUSY_LDST;
                continue;
            }
            // (2) FIFO-occupancy accounting: the in-transit population is the
            // difference of two monotonic counters. (issued - pushed) is the
            // count issued to LDST but not yet in the addr-gen FIFO; adding
            // the current FIFO depth gives the exact count of ops that will
            // occupy a FIFO slot. ldst_ may be null in unit tests => treat the
            // FIFO and the push counter as empty.
            const uint32_t fifo_size =
                ldst_ ? ldst_->current_fifo_size() : 0;
            const uint32_t fifo_pushes =
                ldst_ ? ldst_->current_fifo_total_pushes() : 0;
            const uint32_t fifo_depth =
                ldst_ ? ldst_->current_fifo_capacity() : 0;
            const uint32_t in_flight = ldst_issued_total_ - fifo_pushes;
            if (fifo_depth != 0 && fifo_size + in_flight + 1 > fifo_depth) {
                stats_.warp_stall_unit_busy[w]++;
                stats_.scheduler_ldst_fifo_full_stall_cycles++;
                diagnostics_.next_mut()[w] = SchedulerIssueOutcome::UNIT_BUSY_LDST;
                continue;
            }
        } else {
            // Structural hazard: kUnitIterationLatency is 0 for the fully-
            // pipelined ALU / MULTIPLY, so this never blocks them; armed only
            // for the iterative DIVIDE / TLOOKUP.
            if (unit_busy_.next()[exec_unit_index(target)] > 0) {
                stats_.warp_stall_unit_busy[w]++;
                stats_.scheduler_unit_busy_stall_cycles[exec_unit_index(target)]++;
                diagnostics_.next_mut()[w] = unit_busy_outcome(target);
                continue;
            }
            // Phase 10B.3: the interim writeback-result-buffer gate is gone.
            // The combinational-backward writeback stall now holds a
            // load-preempted fixed-latency result (the stalled unit's gated
            // commit() freezes it), so the scheduler no longer needs to poll
            // each unit's committed result buffer before issuing.
            // Writeback-port hazard: only instructions that write back reserve
            // a slot, so only they can collide.
            if (writes_back && writeback_bitmap_.next()[bitmap_slot(offset)]) {
                stats_.warp_stall_unit_busy[w]++;
                stats_.scheduler_writeback_contention_stall_cycles[
                    exec_unit_index(target)]++;
                diagnostics_.next_mut()[w] = unit_busy_outcome(target);
                continue;
            }
        }

        // Interim operand-collector cooldown — a global single-resource gate
        // (the opcoll is shared by every issue). Drops when opcoll becomes
        // always-1-cycle. Checked after the per-target gates so the per-unit
        // diagnostics above stay precise.
        if (opcoll_cooldown_cycles_.next() > 0) {
            stats_.warp_stall_unit_busy[w]++;
            diagnostics_.next_mut()[w] = SchedulerIssueOutcome::OPCOLL_BUSY;
            continue;
        }

        diagnostics_.next_mut()[w] = SchedulerIssueOutcome::READY_NOT_SELECTED;

        if (!issued) {
            issued = true;
            selected_warp = w;
            selected_entry = entry;
            has_selected_entry = true;
        }
    }

    if (!issued) {
        stats_.scheduler_idle_cycles++;
        if (any_buffer_empty)
            stats_.scheduler_frontend_stall_cycles++;
        else if (any_active)
            stats_.scheduler_stall_backend_cycles++;
    } else if (has_selected_entry) {
        IssueOutput out;
        out.decoded = selected_entry.decoded;
        out.warp_id = selected_warp;
        out.pc = selected_entry.pc;
        out.trace = func_model_.execute(selected_warp, selected_entry.pc);
        out.prediction = selected_entry.prediction;
        output_.set_next(out);

        // Phase 5 of current_mut() elimination (Pattern 2): the pop stays
        // immediate. Scheduler is the last stage in the back-to-front
        // evaluate sweep, so no other stage reads the buffer this cycle
        // after the pop — an immediate pop and a staged pop are
        // observationally equivalent. The push side (decode.stage_push)
        // does stage to align with hardware semantics on the producer's
        // committed-state write. Keeping pop immediate also lets unit-test
        // fixtures (which bypass tick() and so don't call the per-warp
        // buffer.commit) continue to work without modification.
        warps_[selected_warp].instr_buffer.pop();

        // ---- Issue-side bookkeeping for the selected warp ------------------
        const DecodedInstruction& d = selected_entry.decoded;
        const ExecUnit target = d.target_unit;
        const bool is_vdot8   = (d.num_src_regs == 3);
        const bool writes_back = d.has_rd && d.rd != 0;
        const uint32_t offset = issue_to_writeback_offset(target, is_vdot8);

        if (target == ExecUnit::LDST) {
            // LDST reserves no writeback-bitmap slot (variable latency: loads
            // retire via the gather buffer). Advance the issued-total for the
            // FIFO-occupancy gate, and arm the addr-gen structural countdown
            // so the next LDST issue cannot clobber this op mid-addr-gen.
            ldst_issued_total_++;
            if (ldst_iteration_latency_ > 0) {
                unit_busy_.next_mut()[exec_unit_index(ExecUnit::LDST)] =
                    ldst_iteration_latency_;
            }
        } else {
            // Arm the structural-hazard countdown for iterative units.
            if (kUnitIterationLatency[exec_unit_index(target)] > 0) {
                unit_busy_.next_mut()[exec_unit_index(target)] =
                    kUnitIterationLatency[exec_unit_index(target)];
            }
            // Reserve the writeback slot for fixed-latency ops that write back.
            if (writes_back) {
                writeback_bitmap_.next_mut()[bitmap_slot(offset)] = target;
            }
        }
        opcoll_cooldown_cycles_.set_next(is_vdot8 ? 2 : 1);

        if (scoreboard_ != nullptr && writes_back) {
            scoreboard_->set_pending(selected_warp, d.rd);
        }

        diagnostics_.next_mut()[selected_warp] = SchedulerIssueOutcome::ISSUED;
        stats_.total_instructions_issued++;
        stats_.warp_instructions[selected_warp]++;

        if (branch_tracker_ != nullptr &&
            (d.type == InstructionType::BRANCH ||
             d.type == InstructionType::JAL ||
             d.type == InstructionType::JALR)) {
            // Phase 5 REGISTERED: write into next_; visible to scheduler via
            // current_ after commit() flips at end of cycle.
            branch_tracker_->note_branch_issued(selected_warp);
        }
    }

    // RR pointer always advances
    rr_pointer_.next_mut() = (rr_pointer_.next() + 1) % num_warps_;
}

void WarpScheduler::commit() {
    // Phase 10B.3: writeback-stall self-gate. On a stalled cycle the scheduler
    // holds — current_output_ keeps the already-issued instruction (clearing
    // it would lose the issue) and current_diagnostics_ holds. The scoreboard
    // is committed UNCONDITIONALLY by TimingModel (it must carry the arbiter's
    // clear_pending); the scheduler's set_pending is simply absent because the
    // early-returned evaluate() issued nothing this cycle.
    if (wb_arbiter_ != nullptr && wb_arbiter_->next_writeback_stall()) {
        return;
    }
    commit_all();
}

void WarpScheduler::reset() {
    // Phase 4 (reg.h migration): reset_all() T{}-initializes every Reg's
    // current_ and next_ slots — including the writeback_bitmap_ vector,
    // which goes empty. Re-size and re-fill it with the configured length so
    // the in-place mutation in evaluate() addresses a sized buffer (matching
    // the pre-Phase-4 reset that left the vector at construction length).
    reset_all();
    const size_t bitmap_len =
        compute_writeback_bitmap_len(multiply_pipeline_stages_);
    writeback_bitmap_.initialize(std::vector<std::optional<ExecUnit>>(
        bitmap_len, std::nullopt));
    // Phase 10B.0 issue scoreboard. ldst_issued_total_ is cleared in lockstep
    // with LdStUnit::reset()'s fifo_total_pushes_ (both run in the panic-flush
    // cascade) so the (issued - pushed) difference restarts at zero.
    ldst_issued_total_ = 0;
}

void WarpScheduler::flush() {
    reset();
}

} // namespace gpu_sim
