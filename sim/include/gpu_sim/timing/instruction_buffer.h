#pragma once

#include "gpu_sim/timing/branch_predictor.h"
#include "gpu_sim/trace_event.h"
#include <deque>

namespace gpu_sim {

struct BufferEntry {
    DecodedInstruction decoded;
    uint32_t warp_id;
    uint32_t pc;
    BranchPrediction prediction;
};

// Per-warp instruction buffer between decode and the scheduler.
//
// Phase 5 of current_mut() elimination (Pattern 2): decode stages its push
// via stage_push(); TimingModel::tick() iterates warps_ and calls commit()
// per buffer in the commit phase, applying the staged push. This replaces
// the previous decode.commit() pattern that pushed directly into the
// committed deque and then cleared pending_.current_mut() — a Pattern-2
// same-cycle write to committed register state.
//
// Asymmetric discipline: push is staged, pop is immediate. The scheduler
// is the last consumer in the back-to-front evaluate sweep, so an immediate
// pop is observationally equivalent to a staged one. Tests bypass tick()'s
// per-warp commit, so push() (immediate) is used for fixture setup. The
// immediate-pop and flush() entries_.clear() are deferred-Phase-6
// strict-compliance items — both are same-cycle committed-state writes;
// neither is currently expressible without a Wire-mediated handshake that
// the scheduler does not yet have.
class InstructionBuffer {
public:
    explicit InstructionBuffer(uint32_t depth) : max_depth_(depth) {}

    // Committed-state reads.
    bool is_full() const { return entries_.size() >= max_depth_; }
    bool is_empty() const { return entries_.empty(); }
    uint32_t size() const { return static_cast<uint32_t>(entries_.size()); }
    uint32_t capacity() const { return max_depth_; }
    const BufferEntry& front() const { return entries_.front(); }
    BufferEntry& front() { return entries_.front(); }

    // Production: stage a push intent. Applied at commit().
    void stage_push(const BufferEntry& entry) {
        push_ = entry;
        has_push_ = true;
    }

    // Apply the staged push, capacity-checked. Caller (TimingModel tick's
    // commit phase) drives this once per cycle per warp.
    void commit() {
        if (has_push_ && entries_.size() < max_depth_) entries_.push_back(push_);
        has_push_ = false;
    }

    // Same-cycle immediate writes:
    //  - push(): test-setup direct-write to arm initial buffer state.
    //  - pop():  scheduler's same-cycle pop during scheduler.evaluate (the
    //            last stage in the sweep, so no consumer reads the buffer
    //            this cycle after the pop).
    void push(const BufferEntry& entry) {
        if (!is_full()) entries_.push_back(entry);
    }
    void pop() {
        if (!entries_.empty()) entries_.pop_front();
    }

    // Synchronous flush — clears committed entries AND any staged push.
    // Called by fetch.apply_redirect to discard wrong-path state. Same-cycle
    // committed-state write, paired with pop() above as a deferred-Phase-6
    // strict-compliance item.
    void flush() {
        entries_.clear();
        has_push_ = false;
    }
    void reset() { flush(); }

private:
    uint32_t max_depth_;
    std::deque<BufferEntry> entries_;
    BufferEntry push_{};
    bool has_push_ = false;
};

} // namespace gpu_sim
