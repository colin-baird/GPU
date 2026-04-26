#pragma once

#include "gpu_sim/types.h"
#include <array>

namespace gpu_sim {

// Phase 5 REGISTERED state for the per-warp branch-shadow ("branch_in_flight")
// bit. Writers go through one of three event-shaped methods; the bit clears
// on retirement, but the *reason* for retirement (correct prediction vs.
// applied redirect) determines *where* in the tick the clear happens, and
// that placement is load-bearing for correctness — see Phase 5 in the
// timing-discipline doc.
//
// Writers (the only three sites that mutate next_):
//
//   - `note_branch_issued(w)` — `WarpScheduler::evaluate` when it issues a
//     BRANCH/JAL/JALR. Sets next_[w] = true. The scheduler's own next-cycle
//     evaluate reads current_[w] (set by this cycle's commit) and gates
//     subsequent issues for the same warp.
//
//   - `note_resolved_correctly(w)` — `OperandCollector::resolve_branch` when
//     the branch resolves with prediction == actual. Clears next_[w] = false
//     immediately because no flush is pending: the scheduler can resume
//     issuing for warp w on the next cycle.
//
//   - `note_redirect_applied(w)` — `FetchStage::apply_redirect` when a
//     mispredict-redirect actually lands at fetch's commit phase. The
//     OperandCollector deliberately does NOT clear at resolve time for
//     mispredicts, because between resolve and apply (one cycle) the
//     scheduler must still see current_[w] = true so it does not issue a
//     shadow instruction from the soon-to-be-flushed buffer.
//
// All readers (scheduler eligibility check via `is_in_flight`) read current_.
// The scheduler reads current_ before opcoll has had a chance to clear, so a
// branch resolving in cycle N is not visible to the scheduler's evaluate in
// cycle N — it becomes visible in cycle N+1 after commit(). This is exactly
// the Scoreboard pattern (sim/include/gpu_sim/timing/scoreboard.h).
//
// Same-cycle resolution + new-branch-issue invariant: the scheduler can only
// issue a branch for warp W when current_[W] is false. If a branch for W is
// resolving in cycle N (opcoll writes next_[W]=false), the scheduler in
// cycle N still observes current_[W]=true (set by an earlier cycle's commit)
// and therefore won't issue. This means the conflicting "scheduler set +
// opcoll clear" sequence on the same warp slot in next_* during the same
// cycle cannot arise.
class BranchShadowTracker {
public:
    BranchShadowTracker() { reset(); }

    void reset() {
        for (auto& a : current_) a = false;
        for (auto& a : next_) a = false;
    }

    // Reads of committed state.
    bool is_in_flight(WarpId w) const { return current_[w]; }

    // Event-shaped writers. All three write only into next_; commit() flips
    // next_ -> current_ at the cycle boundary.
    void note_branch_issued(WarpId w) { next_[w] = true; }
    void note_resolved_correctly(WarpId w) { next_[w] = false; }
    void note_redirect_applied(WarpId w) { next_[w] = false; }

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
