#include "gpu_sim/timing/warp_scheduler.h"

namespace gpu_sim {

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

    bool issued = false;

    for (uint32_t i = 0; i < num_warps_; ++i) {
        uint32_t w = (rr_pointer_ + i) % num_warps_;

        if (!warps_[w].active) continue;
        if (warps_[w].instr_buffer.is_empty()) {
            stats_.warp_stall_buffer_empty[w]++;
            continue;
        }

        const auto& entry = warps_[w].instr_buffer.front();

        if (!is_scoreboard_clear(w, entry.decoded)) {
            stats_.warp_stall_scoreboard[w]++;
            continue;
        }

        if (!opcoll_free_) {
            continue;
        }

        if (unit_ready_fn_ && !unit_ready_fn_(entry.decoded.target_unit)) {
            stats_.warp_stall_unit_busy[w]++;
            continue;
        }

        // Issue this instruction
        IssueOutput out;
        out.decoded = entry.decoded;
        out.warp_id = w;
        out.pc = entry.pc;

        // Call functional model to execute
        out.trace = func_model_.execute(w, entry.pc);

        next_output_ = out;
        issued = true;

        // Pop from buffer (will take effect at commit)
        warps_[w].instr_buffer.pop();

        // Set scoreboard pending for destination register
        if (entry.decoded.has_rd && entry.decoded.rd != 0) {
            scoreboard_.set_pending(w, entry.decoded.rd);
        }

        stats_.total_instructions_issued++;
        stats_.warp_instructions[w]++;
        break;
    }

    if (!issued) {
        stats_.scheduler_idle_cycles++;
    }

    // RR pointer always advances
    rr_pointer_ = (rr_pointer_ + 1) % num_warps_;
}

void WarpScheduler::commit() {
    current_output_ = next_output_;
}

void WarpScheduler::reset() {
    rr_pointer_ = 0;
    current_output_ = std::nullopt;
    next_output_ = std::nullopt;
    opcoll_free_ = true;
}

} // namespace gpu_sim
