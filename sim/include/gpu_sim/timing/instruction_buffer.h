#pragma once

#include "gpu_sim/timing/branch_predictor.h"
#include "gpu_sim/timing/reg.h"
#include "gpu_sim/trace_event.h"

namespace gpu_sim {

struct BufferEntry {
    DecodedInstruction decoded;
    uint32_t warp_id;
    uint32_t pc;
    BranchPrediction prediction;
};

// Per-warp instruction buffer between decode and the scheduler.
//
// Phase 6 (sparkling-dazzling-starfish.md) — strict-compliance discipline:
// the internal storage is a `RegFifo<BufferEntry>`. Push, pop, and flush are
// all staged in evaluate() and applied at the per-warp `commit()` driven from
// `TimingModel::tick()`'s commit phase. This replaces the prior asymmetric
// shape (staged push + immediate pop + immediate flush() from
// FetchStage::apply_redirect()) — both pop and flush are now staged, the
// flush via a `Wire<std::array<bool, MAX_WARPS>> instr_buffer_flush_request_`
// owned by `TimingModel` and driven by `FetchStage::apply_redirect()`.
//
// On a cycle where decode stages a push, scheduler stages a pop, and fetch
// drives the flush wire for this warp: the flush wins (the redirected warp's
// buffer is cleared and any staged push/pop discarded). Without a flush,
// commit() applies pop-then-push atomically via the underlying RegFifo
// commit semantics. The decode-side `is_full()` check reads the committed
// size start-of-cycle (no same-cycle pop credit), preserving the prior
// fetch-side conservatism.
//
// Test-only convenience helpers `push()` / `pop()` / `flush()` perform an
// immediate apply (stage + commit on the underlying RegFifo, or reset) so
// fixture setup that pre-arms a buffer without driving the per-warp commit
// still works. The production callers (`FetchStage::apply_redirect`,
// `WarpScheduler::evaluate`, `DecodeStage::evaluate`) all use the
// stage-and-defer path.
class InstructionBuffer {
public:
    explicit InstructionBuffer(uint32_t depth) : max_depth_(depth) {}

    // Committed-state reads — start-of-cycle / post-commit.
    bool is_full() const { return entries_.current_size() >= max_depth_; }
    bool is_empty() const { return entries_.current_empty(); }
    uint32_t size() const { return static_cast<uint32_t>(entries_.current_size()); }
    uint32_t capacity() const { return max_depth_; }
    const BufferEntry& front() const { return entries_.current_front(); }

    // Production: stage a push intent. Applied at commit() (subject to a
    // same-cycle flush request, which discards it). Capacity-checked
    // against committed size + already-staged pushes; over-cap pushes are
    // silently dropped (the decode-side `is_full()` check normally gates
    // this, but for robustness the commit applies min(staged, capacity)).
    void stage_push(const BufferEntry& entry) {
        if (entries_.current_size() + entries_.pushes_staged() < max_depth_) {
            entries_.stage_push(entry);
        }
    }

    // Production: stage a pop intent. Applied at commit().
    void stage_pop() { entries_.stage_pop(); }

    // Apply staged ops (or honor a flush request). Caller (TimingModel
    // tick's commit phase, once per cycle per warp) drives this. A flush
    // request clears the committed queue and discards both staged pushes
    // and staged pops — the redirected warp's wrong-path state is gone.
    void commit(bool flush_request = false) {
        if (flush_request) {
            entries_.reset();
            return;
        }
        entries_.commit();
    }

    // Test-only convenience: immediate push (stage + apply). Fixture setup
    // uses this to pre-arm a buffer without driving the per-warp commit
    // from a tick(). Production callers must use stage_push().
    void push(const BufferEntry& entry) {
        if (entries_.current_size() < max_depth_) {
            entries_.stage_push(entry);
            entries_.commit();
        }
    }

    // Test-only convenience: immediate pop (stage + apply).
    void pop() {
        if (!entries_.current_empty()) {
            entries_.stage_pop();
            entries_.commit();
        }
    }

    // Test-only convenience: immediate flush. Production callers must drive
    // the per-warp flush Wire owned by TimingModel; FetchStage::apply_redirect
    // does so. The deferred commit honors the wire's value.
    void flush() { entries_.reset(); }
    void reset() { entries_.reset(); }

private:
    uint32_t max_depth_;  // config
    // InstructionBuffer does not derive RegisteredStage: WarpState owns one
    // InstructionBuffer per warp and is stored in a std::vector<WarpState>
    // (see warp_state.h). The class must remain movable for the vector's
    // emplace_back pattern used by TimingModel and every test fixture.
    // The buffer is instead driven by an explicit commit(bool flush_request)
    // method called once per cycle per warp from TimingModel::tick()'s
    // commit phase (parallels the WarpState::commit() helper for pc_ /
    // active_). Lint state-shape exception documented here.
    RegFifo<BufferEntry> entries_;  // timing-naming-allow: driven via commit(bool) helper called from TimingModel::tick()'s per-warp commit phase; InstructionBuffer stays non-RegisteredStage to remain movable for std::vector<WarpState>.
};

} // namespace gpu_sim
