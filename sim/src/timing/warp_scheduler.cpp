#include "gpu_sim/timing/warp_scheduler.h"

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
                             Scoreboard& scoreboard, FunctionalModel& func_model,
                             Stats& stats)
    : num_warps_(num_warps), warps_(warps), scoreboard_(scoreboard),
      func_model_(func_model), stats_(stats) {}

bool WarpScheduler::is_scoreboard_clear(WarpId warp, const DecodedInstruction& d) const {
    // Check rs1
    if (d.num_src_regs >= 1 && scoreboard_.is_pending(warp, d.rs1)) return false;
    // Check rs2 (for 2+ operand instructions)
    if (d.num_src_regs >= 2 && scoreboard_.is_pending(warp, d.rs2)) return false;
    // Check rd as source (VDOT8: 3-operand)
    if (d.reads_rd && scoreboard_.is_pending(warp, d.rd)) return false;
    return true;
}

void WarpScheduler::evaluate() {
    next_output_ = std::nullopt;
    next_diagnostics_.fill(SchedulerIssueOutcome::INACTIVE);

    bool issued = false;
    uint32_t selected_warp = 0;
    BufferEntry selected_entry{};
    bool has_selected_entry = false;

    for (uint32_t i = 0; i < num_warps_; ++i) {
        uint32_t w = (rr_pointer_ + i) % num_warps_;

        if (!warps_[w].active) {
            next_diagnostics_[w] = SchedulerIssueOutcome::INACTIVE;
            continue;
        }

        if (warps_[w].instr_buffer.is_empty()) {
            stats_.warp_stall_buffer_empty[w]++;
            next_diagnostics_[w] = SchedulerIssueOutcome::BUFFER_EMPTY;
            continue;
        }

        const auto& entry = warps_[w].instr_buffer.front();

        if (!is_scoreboard_clear(w, entry.decoded)) {
            stats_.warp_stall_scoreboard[w]++;
            next_diagnostics_[w] = SchedulerIssueOutcome::SCOREBOARD;
            continue;
        }

        if (!opcoll_free_) {
            stats_.warp_stall_unit_busy[w]++;
            next_diagnostics_[w] = SchedulerIssueOutcome::OPCOLL_BUSY;
            continue;
        }

        if (unit_ready_fn_ && !unit_ready_fn_(entry.decoded.target_unit)) {
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
    opcoll_free_ = true;
    current_diagnostics_.fill(SchedulerIssueOutcome::INACTIVE);
    next_diagnostics_.fill(SchedulerIssueOutcome::INACTIVE);
}

} // namespace gpu_sim
