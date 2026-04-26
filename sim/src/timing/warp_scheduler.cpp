#include "gpu_sim/timing/warp_scheduler.h"

#include "gpu_sim/timing/operand_collector.h"
#include "gpu_sim/timing/alu_unit.h"
#include "gpu_sim/timing/multiply_unit.h"
#include "gpu_sim/timing/divide_unit.h"
#include "gpu_sim/timing/tlookup_unit.h"
#include "gpu_sim/timing/ldst_unit.h"

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
                             Scoreboard& scoreboard,
                             BranchShadowTracker& branch_tracker,
                             FunctionalModel& func_model, Stats& stats)
    : num_warps_(num_warps), warps_(warps), scoreboard_(scoreboard),
      branch_tracker_(branch_tracker), func_model_(func_model), stats_(stats) {}

bool WarpScheduler::is_scoreboard_clear(WarpId warp, const DecodedInstruction& d) const {
    // Check rs1
    if (d.num_src_regs >= 1 && scoreboard_.is_pending(warp, d.rs1)) return false;
    // Check rs2 (for 2+ operand instructions)
    if (d.num_src_regs >= 2 && scoreboard_.is_pending(warp, d.rs2)) return false;
    // Check rd as source (VDOT8: 3-operand)
    if (d.reads_rd && scoreboard_.is_pending(warp, d.rd)) return false;
    return true;
}

bool WarpScheduler::query_opcoll_ready() const {
    // Phase 4 READY/STALL: prefer test override, then wired opcoll's
    // ready_out() (a const accessor over committed state) during
    // scheduler.evaluate() in TimingModel::tick(). Fallback for tests that
    // wire neither: free.
    if (opcoll_ready_override_) return *opcoll_ready_override_;
    if (opcoll_) return opcoll_->ready_out();
    return true;
}

bool WarpScheduler::query_unit_ready(ExecUnit unit) const {
    auto resolve = [](const std::optional<bool>& override_,
                      const ExecutionUnit* wired) -> bool {
        if (override_) return *override_;
        if (wired) return wired->ready_out();
        return true;
    };
    switch (unit) {
        case ExecUnit::ALU:      return resolve(alu_ready_override_, alu_);
        case ExecUnit::MULTIPLY: return resolve(mul_ready_override_, mul_);
        case ExecUnit::DIVIDE:   return resolve(div_ready_override_, div_);
        case ExecUnit::TLOOKUP:  return resolve(tlookup_ready_override_, tlookup_);
        case ExecUnit::LDST:     return resolve(ldst_ready_override_, ldst_);
        case ExecUnit::SYSTEM:   return true;
        default:                 return false;
    }
}

void WarpScheduler::evaluate() {
    next_output_ = std::nullopt;
    next_diagnostics_.fill(SchedulerIssueOutcome::INACTIVE);

    bool issued = false;
    bool any_buffer_empty = false;
    bool any_active = false;
    uint32_t selected_warp = 0;
    BufferEntry selected_entry{};
    bool has_selected_entry = false;

    // Phase 4 READY/STALL: query opcoll once per evaluate. ready_out() reads
    // only committed (current_busy_) state, so the value is stable for the
    // entire cycle and does not require a separate pre-pass.
    const bool opcoll_ready = query_opcoll_ready();

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

        // Phase 5 REGISTERED: read tracker.is_in_flight(w) which exposes
        // current_ (committed) state. The note_branch_issued write below
        // goes into next_; commit() flips at end of cycle.
        if (branch_tracker_.is_in_flight(w)) {
            stats_.warp_stall_branch_shadow[w]++;
            next_diagnostics_[w] = SchedulerIssueOutcome::BRANCH_SHADOW;
            continue;
        }

        const auto& entry = warps_[w].instr_buffer.front();

        if (!is_scoreboard_clear(w, entry.decoded)) {
            stats_.warp_stall_scoreboard[w]++;
            next_diagnostics_[w] = SchedulerIssueOutcome::SCOREBOARD;
            continue;
        }

        if (!opcoll_ready) {
            stats_.warp_stall_unit_busy[w]++;
            next_diagnostics_[w] = SchedulerIssueOutcome::OPCOLL_BUSY;
            continue;
        }

        if (!query_unit_ready(entry.decoded.target_unit)) {
            stats_.warp_stall_unit_busy[w]++;
            next_diagnostics_[w] = unit_busy_outcome(entry.decoded.target_unit);
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
        if (selected_entry.decoded.has_rd && selected_entry.decoded.rd != 0) {
            scoreboard_.set_pending(selected_warp, selected_entry.decoded.rd);
        }

        next_diagnostics_[selected_warp] = SchedulerIssueOutcome::ISSUED;
        stats_.total_instructions_issued++;
        stats_.warp_instructions[selected_warp]++;

        if (selected_entry.decoded.type == InstructionType::BRANCH ||
            selected_entry.decoded.type == InstructionType::JAL ||
            selected_entry.decoded.type == InstructionType::JALR) {
            // Phase 5 REGISTERED: write into next_; visible to scheduler via
            // current_ after commit() flips at end of cycle.
            branch_tracker_.note_branch_issued(selected_warp);
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
}

void WarpScheduler::flush() {
    reset();
}

} // namespace gpu_sim
