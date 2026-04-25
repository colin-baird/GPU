#pragma once

#include "gpu_sim/types.h"
#include <array>

namespace gpu_sim {

// Phase 5 REGISTERED state for the per-warp branch-shadow ("branch_in_flight")
// bit. Two writers (scheduler.evaluate sets on issue of a BRANCH/JAL/JALR;
// opcoll branch-resolution clears on retirement) write only into next_*.
// All readers (scheduler eligibility check) read current_. The scheduler
// reads current_ before opcoll has had a chance to clear, so a branch
// resolving in cycle N is not visible to the scheduler's evaluate in cycle
// N — it becomes visible in cycle N+1 after commit(). This is exactly the
// Scoreboard pattern (sim/include/gpu_sim/timing/scoreboard.h).
//
// Same-cycle resolution + new-branch-issue invariant: the scheduler can
// only issue a branch for warp W when current_[W] is false. If a branch
// for W is resolving in cycle N (opcoll writes next_[W]=false), the
// scheduler in cycle N still observes current_[W]=true (set by an earlier
// cycle's commit) and therefore won't issue. This means the conflicting
// "scheduler set + opcoll clear" sequence on the same warp slot in next_*
// during the same cycle cannot arise. Verified by reading
// warp_scheduler.cpp's BRANCH_SHADOW gate, which sits ahead of the issue
// and reads current_.
class BranchShadowTracker {
public:
    BranchShadowTracker() { reset(); }

    void reset() {
        for (auto& a : current_) a = false;
        for (auto& a : next_) a = false;
    }

    // Reads of committed state.
    bool is_in_flight(WarpId w) const { return current_[w]; }

    // Writes go to next_ only.
    void set_in_flight(WarpId w) { next_[w] = true; }
    void clear_in_flight(WarpId w) { next_[w] = false; }

    // Seed next_ from current_ at the top of each cycle so unmodified slots
    // carry forward.
    void seed_next() {
        for (uint32_t i = 0; i < MAX_WARPS; ++i) next_[i] = current_[i];
    }

    // Flip double-buffered state at the cycle boundary.
    void commit() {
        for (uint32_t i = 0; i < MAX_WARPS; ++i) current_[i] = next_[i];
    }

private:
    std::array<bool, MAX_WARPS> current_{};
    std::array<bool, MAX_WARPS> next_{};
};

} // namespace gpu_sim
