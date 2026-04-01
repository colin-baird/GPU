#include "gpu_sim/timing/writeback_arbiter.h"

namespace gpu_sim {

WritebackArbiter::WritebackArbiter(Scoreboard& scoreboard, Stats& stats)
    : scoreboard_(scoreboard), stats_(stats) {}

void WritebackArbiter::add_source(ExecutionUnit* unit) {
    sources_.push_back(unit);
}

void WritebackArbiter::evaluate() {
    pending_commit_ = std::nullopt;

    uint32_t valid_count = 0;
    int32_t winner = -1;

    for (uint32_t i = 0; i < sources_.size(); ++i) {
        uint32_t idx = (rr_pointer_ + i) % static_cast<uint32_t>(sources_.size());
        if (sources_[idx]->has_result()) {
            valid_count++;
            if (winner < 0) {
                winner = static_cast<int32_t>(idx);
            }
        }
    }

    if (valid_count > 1) {
        stats_.writeback_conflicts++;
    }

    if (winner >= 0) {
        uint32_t idx = static_cast<uint32_t>(winner);
        WritebackEntry entry = sources_[idx]->consume_result();

        pending_commit_ = entry;
        rr_pointer_ = (idx + 1) % static_cast<uint32_t>(sources_.size());

        if (entry.dest_reg != 0) {
            scoreboard_.clear_pending(entry.warp_id, entry.dest_reg);
        }
    }
}

void WritebackArbiter::commit() {
    committed_ = pending_commit_;
}

bool WritebackArbiter::has_pending_work() const {
    if (pending_commit_.has_value()) {
        return true;
    }

    for (const auto* source : sources_) {
        if (source->has_result()) {
            return true;
        }
    }

    return false;
}

uint32_t WritebackArbiter::ready_source_count() const {
    uint32_t count = 0;
    for (const auto* source : sources_) {
        if (source->has_result()) {
            count++;
        }
    }
    return count;
}

void WritebackArbiter::reset() {
    rr_pointer_ = 0;
    committed_ = std::nullopt;
    pending_commit_ = std::nullopt;
}

} // namespace gpu_sim
