#include "gpu_sim/timing/warp_scheduler.h"

#include "gpu_sim/timing/ldst_unit.h"
#include "gpu_sim/timing/alu_unit.h"
#include "gpu_sim/timing/multiply_unit.h"
#include "gpu_sim/timing/divide_unit.h"
#include "gpu_sim/timing/tlookup_unit.h"

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
                             FunctionalModel& func_model, Stats& stats)
    : num_warps_(num_warps), warps_(warps),
      func_model_(func_model), stats_(stats) {}

void WarpScheduler::set_dependencies(Scoreboard* scoreboard,
                                     BranchShadowTracker* branch_tracker,
                                     LdStUnit* ldst, ALUUnit* alu,
                                     MultiplyUnit* mul, DivideUnit* div,
                                     TLookupUnit* tlookup) {
    scoreboard_ = scoreboard;
    branch_tracker_ = branch_tracker;
    ldst_ = ldst;
    alu_ = alu;
    mul_ = mul;
    div_ = div;
    tlookup_ = tlookup;
    // Capture the LDST addr-gen iteration latency (runtime-configured value;
    // see warp_scheduler.h). Defined here rather than inline in the header so
    // LdStUnit stays an incomplete type at the header.
    ldst_iteration_latency_ = ldst_ ? ldst_->current_addr_gen_latency() : 0;
}

// Phase 10B.0 interim writeback-result-buffer gate (human-approved deviation,
// removed in 10B.3). Returns true when the target fixed-latency unit's
// committed result buffer is still occupied by an unconsumed result — issuing
// a new op into it would let the unit's next-cycle evaluate() overwrite that
// result (ALU/DIV/TLOOKUP) or stall its pipeline (MUL). The narrow
// current_result_buffer_.valid portion of the old current_busy(). LDST/SYSTEM
// produce no fixed-latency writeback and are never gated here.
//
// A member function (not a free function) so each unit->scheduler
// current_result_pending() read has a statically resolvable receiver field
// (alu_ / mul_ / div_ / tlookup_) — the libclang AST extractor in
// tools/diagram_extract_ast.py attributes the cross-stage edge correctly, the
// same reason the 10A-era query_unit_ready() was inlined per-case rather than
// hidden behind a pointer-parameter lambda.
bool WarpScheduler::target_result_buffer_occupied(ExecUnit target) const {
    switch (target) {
        case ExecUnit::ALU:
            return alu_ != nullptr && alu_->current_result_pending();
        case ExecUnit::MULTIPLY:
            return mul_ != nullptr && mul_->current_result_pending();
        case ExecUnit::DIVIDE:
            return div_ != nullptr && div_->current_result_pending();
        case ExecUnit::TLOOKUP:
            return tlookup_ != nullptr && tlookup_->current_result_pending();
        default:
            return false;
    }
}

bool WarpScheduler::is_scoreboard_clear(WarpId warp, const DecodedInstruction& d) const {
    // No scoreboard wired (unit-test default): no register hazards.
    if (scoreboard_ == nullptr) return true;
    if (d.num_src_regs >= 1 && scoreboard_->current_pending(warp, d.rs1)) return false;
    if (d.num_src_regs >= 2 && scoreboard_->current_pending(warp, d.rs2)) return false;
    // Check rd as source (VDOT8: 3-operand)
    if (d.reads_rd && scoreboard_->current_pending(warp, d.rd)) return false;
    return true;
}

void WarpScheduler::evaluate() {
    next_output_ = std::nullopt;
    next_diagnostics_.fill(SchedulerIssueOutcome::INACTIVE);

    // ---- Top-of-evaluate issue-scoreboard bookkeeping ----------------------
    // Phase 10B.0: the scheduler predicts unit availability from issue history
    // rather than polling its consumers. This bookkeeping runs once per cycle,
    // before the issue gate reads any of it, so a value set on issue at cycle
    // N is checked one lower at N+1 (the decrement-then-check ordering that
    // makes the countdowns exact).
    //
    // 10B.3 will add a writeback-stall freeze guard here: on a stalled cycle
    // the scheduler early-returns at the top so bitmap_head_ does not advance
    // and no issue bookkeeping moves — the schedule simply pauses for a cycle.
    writeback_bitmap_[bitmap_head_] = std::nullopt;   // reserved cycle elapsed
    bitmap_head_ = (bitmap_head_ + 1) % kWritebackBitmapLen;
    for (auto& b : unit_busy_) {
        if (b > 0) --b;                                // DIVIDE / TLOOKUP only
    }
    if (opcoll_cooldown_cycles_ > 0) {
        --opcoll_cooldown_cycles_;
    }

    bool issued = false;
    bool any_buffer_empty = false;
    bool any_active = false;
    uint32_t selected_warp = 0;
    BufferEntry selected_entry{};
    bool has_selected_entry = false;

    for (uint32_t i = 0; i < num_warps_; ++i) {
        uint32_t w = (rr_pointer_ + i) % num_warps_;

        if (!warps_[w].active) {
            next_diagnostics_[w] = SchedulerIssueOutcome::INACTIVE;
            continue;
        }
        any_active = true;

        if (warps_[w].instr_buffer.is_empty()) {
            stats_.warp_stall_buffer_empty[w]++;
            next_diagnostics_[w] = SchedulerIssueOutcome::BUFFER_EMPTY;
            any_buffer_empty = true;
            continue;
        }

        // Phase 5 REGISTERED: read tracker.current_in_flight(w) which exposes
        // current_ (committed) state. The note_branch_issued write below
        // goes into next_; commit() flips at end of cycle. No tracker
        // wired (unit-test default): no branch shadow ever in flight.
        if (branch_tracker_ != nullptr && branch_tracker_->current_in_flight(w)) {
            stats_.warp_stall_branch_shadow[w]++;
            next_diagnostics_[w] = SchedulerIssueOutcome::BRANCH_SHADOW;
            continue;
        }

        const auto& entry = warps_[w].instr_buffer.front();
        const DecodedInstruction& decoded = entry.decoded;

        if (!is_scoreboard_clear(w, decoded)) {
            stats_.warp_stall_scoreboard[w]++;
            next_diagnostics_[w] = SchedulerIssueOutcome::SCOREBOARD;
            continue;
        }

        // Phase 10B.0 combined issue gate ------------------------------------
        const ExecUnit target = decoded.target_unit;
        const bool is_vdot8   = (decoded.num_src_regs == 3);
        const bool writes_back = decoded.has_rd && decoded.rd != 0;
        // offset is read by both the bitmap conflict check and the reservation
        // at issue. Harmless for LDST / SYSTEM (offset 0, never reserve).
        const uint32_t offset = compute_issue_to_writeback_offset(target, is_vdot8);

        if (target == ExecUnit::LDST) {
            // (1) Addr-gen structural hazard. LdStUnit's address-generation
            // stage holds one op at a time and accept() unconditionally
            // overwrites it, so consecutive LDST issues must be spaced by the
            // addr-gen latency or the second clobbers an in-flight op. The
            // unit_busy_[LDST] countdown enforces that spacing.
            if (unit_busy_[exec_unit_index(ExecUnit::LDST)] > 0) {
                stats_.warp_stall_unit_busy[w]++;
                stats_.scheduler_unit_busy_stall_cycles[
                    exec_unit_index(ExecUnit::LDST)]++;
                next_diagnostics_[w] = SchedulerIssueOutcome::UNIT_BUSY_LDST;
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
                next_diagnostics_[w] = SchedulerIssueOutcome::UNIT_BUSY_LDST;
                continue;
            }
        } else {
            // Structural hazard: kUnitIterationLatency is 0 for the fully-
            // pipelined ALU / MULTIPLY, so this never blocks them; armed only
            // for the iterative DIVIDE / TLOOKUP.
            if (unit_busy_[exec_unit_index(target)] > 0) {
                stats_.warp_stall_unit_busy[w]++;
                stats_.scheduler_unit_busy_stall_cycles[exec_unit_index(target)]++;
                next_diagnostics_[w] = unit_busy_outcome(target);
                continue;
            }
            // Phase 10B.0 INTERIM writeback-result-buffer gate (DELIBERATE,
            // HUMAN-APPROVED DEVIATION from the plan — removed in 10B.3).
            // The plan's 10B.0 drops the scheduler's current_busy() poll, but
            // that poll was doing double duty: structural-input hazard (now
            // the unit_busy_/bitmap bookkeeping) AND "committed result buffer
            // still occupied". The bitmap only prevents fixed-vs-fixed
            // collisions at issue; the 10B.0 arbiter is still round-robin and
            // ignores the bitmap, so a load can preempt a fixed-latency
            // writeback and leave an unconsumed result that the unit's next
            // evaluate() would clobber (ALU/DIV/TLOOKUP overwrite next_result_
            // buffer_) -> lost writeback -> scoreboard destination never
            // cleared -> dependent warps deadlock. Until 10B.3's fixed-
            // priority arbiter + combinational-backward writeback stall holds
            // a preempted result, the scheduler must not issue into a unit
            // whose committed result buffer is still occupied. This is the
            // narrow current_result_buffer_.valid portion of the old
            // current_busy() — a backward committed-state back-pressure read,
            // discipline-compliant. It is unit-level (not gated on
            // writes_back): even a non-writeback ALU op deposits a result-
            // buffer entry the arbiter must drain. Removed in 10B.3; Phase
            // 10F's doc sweep records the deviation.
            if (target_result_buffer_occupied(target)) {
                stats_.warp_stall_unit_busy[w]++;
                stats_.scheduler_unit_busy_stall_cycles[exec_unit_index(target)]++;
                next_diagnostics_[w] = unit_busy_outcome(target);
                continue;
            }
            // Writeback-port hazard: only instructions that write back reserve
            // a slot, so only they can collide.
            if (writes_back &&
                writeback_bitmap_[(bitmap_head_ + offset) % kWritebackBitmapLen]) {
                stats_.warp_stall_unit_busy[w]++;
                stats_.scheduler_writeback_contention_stall_cycles[
                    exec_unit_index(target)]++;
                next_diagnostics_[w] = unit_busy_outcome(target);
                continue;
            }
        }

        // Interim operand-collector cooldown — a global single-resource gate
        // (the opcoll is shared by every issue). Drops when opcoll becomes
        // always-1-cycle. Checked after the per-target gates so the per-unit
        // diagnostics above stay precise.
        if (opcoll_cooldown_cycles_ > 0) {
            stats_.warp_stall_unit_busy[w]++;
            next_diagnostics_[w] = SchedulerIssueOutcome::OPCOLL_BUSY;
            continue;
        }

        next_diagnostics_[w] = SchedulerIssueOutcome::READY_NOT_SELECTED;

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
        next_output_ = out;

        warps_[selected_warp].instr_buffer.pop();

        // ---- Issue-side bookkeeping for the selected warp ------------------
        const DecodedInstruction& d = selected_entry.decoded;
        const ExecUnit target = d.target_unit;
        const bool is_vdot8   = (d.num_src_regs == 3);
        const bool writes_back = d.has_rd && d.rd != 0;
        const uint32_t offset = compute_issue_to_writeback_offset(target, is_vdot8);

        if (target == ExecUnit::LDST) {
            // LDST reserves no writeback-bitmap slot (variable latency: loads
            // retire via the gather buffer). Advance the issued-total for the
            // FIFO-occupancy gate, and arm the addr-gen structural countdown
            // so the next LDST issue cannot clobber this op mid-addr-gen.
            ldst_issued_total_++;
            if (ldst_iteration_latency_ > 0) {
                unit_busy_[exec_unit_index(ExecUnit::LDST)] =
                    ldst_iteration_latency_;
            }
        } else {
            // Arm the structural-hazard countdown for iterative units.
            if (kUnitIterationLatency[exec_unit_index(target)] > 0) {
                unit_busy_[exec_unit_index(target)] =
                    kUnitIterationLatency[exec_unit_index(target)];
            }
            // Reserve the writeback slot for fixed-latency ops that write back.
            if (writes_back) {
                writeback_bitmap_[(bitmap_head_ + offset) % kWritebackBitmapLen] =
                    target;
            }
        }
        opcoll_cooldown_cycles_ = is_vdot8 ? 2 : 1;

        if (scoreboard_ != nullptr && writes_back) {
            scoreboard_->set_pending(selected_warp, d.rd);
        }

        next_diagnostics_[selected_warp] = SchedulerIssueOutcome::ISSUED;
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
    rr_pointer_ = (rr_pointer_ + 1) % num_warps_;
}

void WarpScheduler::commit() {
    current_output_ = next_output_;
    current_diagnostics_ = next_diagnostics_;
}

void WarpScheduler::reset() {
    rr_pointer_ = 0;
    current_output_ = std::nullopt;
    next_output_ = std::nullopt;
    current_diagnostics_.fill(SchedulerIssueOutcome::INACTIVE);
    next_diagnostics_.fill(SchedulerIssueOutcome::INACTIVE);
    // Phase 10B.0 issue scoreboard. ldst_issued_total_ is cleared in lockstep
    // with LdStUnit::reset()'s fifo_total_pushes_ (both run in the panic-flush
    // cascade) so the (issued - pushed) difference restarts at zero.
    unit_busy_.fill(0);
    writeback_bitmap_.fill(std::nullopt);
    bitmap_head_ = 0;
    ldst_issued_total_ = 0;
    opcoll_cooldown_cycles_ = 0;
}

void WarpScheduler::flush() {
    reset();
}

} // namespace gpu_sim
