#pragma once

#include "gpu_sim/timing/reg.h"
#include "gpu_sim/types.h"

namespace gpu_sim {

// POD wrapping the whole per-warp register-pending bit array as one value, so
// it can be held in a single Reg<T> (whole-array wrapping — not per-element
// Reg<bool>). A Reg<ScoreboardBits> copy compiles to the same array copy the
// hand-rolled double-buffer did.
struct ScoreboardBits {
    bool pending[MAX_WARPS][NUM_REGS] = {};
};

class Scoreboard : public RegisteredStage {
public:
    Scoreboard() {
        register_state(&bits_);
        reset();
    }

    // Reset BOTH committed and staged state (Reg::reset() clears both slots).
    void reset() { reset_all(); }

    // Read from current state (used by scheduler)
    bool current_pending(WarpId warp, RegIndex reg) const {
        if (reg == 0) return false;
        return bits_.current().pending[warp][reg];
    }

    // Write to next state (used by issue and writeback)
    void set_pending(WarpId warp, RegIndex reg) {
        if (reg == 0) return;
        bits_.next_mut().pending[warp][reg] = true;
    }

    void clear_pending(WarpId warp, RegIndex reg) {
        if (reg == 0) return;
        bits_.next_mut().pending[warp][reg] = false;
    }

    // Flip double-buffered state.
    void commit() { commit_all(); }

    // Seed next from current at start of cycle.
    void seed_next() { seed_all(); }

private:
    Reg<ScoreboardBits> bits_;
};

} // namespace gpu_sim
