#pragma once

#include "gpu_sim/types.h"
#include <cstring>

namespace gpu_sim {

class Scoreboard {
public:
    Scoreboard() { reset(); }

    void reset() {
        std::memset(current_, 0, sizeof(current_));
        std::memset(next_, 0, sizeof(next_));
    }

    // Read from current state (used by scheduler)
    bool is_pending(WarpId warp, RegIndex reg) const {
        if (reg == 0) return false;
        return current_[warp][reg];
    }

    // Write to next state (used by issue and writeback)
    void set_pending(WarpId warp, RegIndex reg) {
        if (reg == 0) return;
        next_[warp][reg] = true;
    }

    void clear_pending(WarpId warp, RegIndex reg) {
        if (reg == 0) return;
        next_[warp][reg] = false;
    }

    // Flip double-buffered state
    void commit() {
        std::memcpy(current_, next_, sizeof(current_));
    }

    // Seed next from current at start of cycle
    void seed_next() {
        std::memcpy(next_, current_, sizeof(next_));
    }

private:
    bool current_[MAX_WARPS][NUM_REGS];
    bool next_[MAX_WARPS][NUM_REGS];
};

} // namespace gpu_sim
