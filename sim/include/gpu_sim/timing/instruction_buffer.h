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
// Phase 5 of current_mut() elimination (Pattern 2): the buffer now has a
// RegFifo-style commit-disciplined API on top of the underlying deque.
// Production code stages pushes (decode.evaluate) and pops (scheduler.evaluate)
// via stage_push() / stage_pop(); TimingModel::tick() iterates warps_ and
// calls commit() per buffer in the commit phase, which applies pop-then-push.
// This replaces the previous decode.commit() pattern that pushed directly into
// the committed deque and then cleared pending_.current_mut() — a Pattern-2
// same-cycle write to committed register state.
//
// Tests still use push() / pop() as immediate-write helpers for fixture setup;
// production code uses the staged API.
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

    // Production: stage a push / pop intent. Applied at commit().
    void stage_push(const BufferEntry& entry) {
        push_ = entry;
        has_push_ = true;
    }
    void stage_pop() { pop_ = true; }

    // Apply staged ops: pop-then-push, capacity-checked. Caller (TimingModel
    // tick's commit phase) drives this once per cycle per warp.
    void commit() {
        if (pop_ && !entries_.empty()) entries_.pop_front();
        if (has_push_ && entries_.size() < max_depth_) entries_.push_back(push_);
        has_push_ = false;
        pop_ = false;
    }

    // Test-setup immediate writes — bypass staging. Used by tests to arm
    // initial buffer state.
    void push(const BufferEntry& entry) {
        if (!is_full()) entries_.push_back(entry);
    }
    void pop() {
        if (!entries_.empty()) entries_.pop_front();
    }

    // Synchronous flush — clears committed entries AND any staged ops.
    // Called by fetch.apply_redirect to discard wrong-path state. (The
    // direct entries_.clear() is itself a same-cycle committed-state write
    // and is a candidate for future strict-compliance cleanup, but is out
    // of scope for the Pattern-2 elimination.)
    void flush() {
        entries_.clear();
        has_push_ = false;
        pop_ = false;
    }
    void reset() { flush(); }

private:
    uint32_t max_depth_;
    std::deque<BufferEntry> entries_;
    BufferEntry push_{};
    bool has_push_ = false;
    bool pop_ = false;
};

} // namespace gpu_sim
