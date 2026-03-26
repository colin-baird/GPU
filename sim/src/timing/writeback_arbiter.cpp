#include "gpu_sim/timing/writeback_arbiter.h"

namespace gpu_sim {

WritebackArbiter::WritebackArbiter(Scoreboard& scoreboard, Stats& stats)
    : scoreboard_(scoreboard), stats_(stats) {}

void WritebackArbiter::add_source(ExecutionUnit* unit) {
    sources_.push_back(unit);
}

void WritebackArbiter::submit_fill(const WritebackEntry& entry) {
    fill_buffer_ = entry;
}

void WritebackArbiter::evaluate() {
    pending_commit_ = std::nullopt;

    uint32_t total_sources = static_cast<uint32_t>(sources_.size()) + 1;  // +1 for fill buffer
    uint32_t valid_count = 0;
    int32_t winner = -1;

    // Phase 1: Count valid sources and find winner (round-robin)
    for (uint32_t i = 0; i < total_sources; ++i) {
        uint32_t idx = (rr_pointer_ + i) % total_sources;
        bool has_result = false;

        if (idx < sources_.size()) {
            has_result = sources_[idx]->has_result();
        } else {
            has_result = fill_buffer_.valid;
        }

        if (has_result) {
            valid_count++;
            if (winner < 0) {
                winner = static_cast<int32_t>(idx);
            }
        }
    }

    if (valid_count > 1) {
        stats_.writeback_conflicts++;
    }

    // Phase 2: Consume only the winner
    if (winner >= 0) {
        uint32_t idx = static_cast<uint32_t>(winner);
        WritebackEntry entry;

        if (idx < sources_.size()) {
            entry = sources_[idx]->consume_result();
        } else {
            entry = fill_buffer_;
            fill_buffer_.valid = false;
        }

        pending_commit_ = entry;
        rr_pointer_ = (idx + 1) % total_sources;

        if (entry.dest_reg != 0) {
            scoreboard_.clear_pending(entry.warp_id, entry.dest_reg);
        }
    }
}

void WritebackArbiter::commit() {
    committed_ = pending_commit_;
}

void WritebackArbiter::reset() {
    rr_pointer_ = 0;
    committed_ = std::nullopt;
    pending_commit_ = std::nullopt;
    fill_buffer_.valid = false;
}

} // namespace gpu_sim
