#pragma once

#include "gpu_sim/trace_event.h"
#include <deque>

namespace gpu_sim {

struct BufferEntry {
    DecodedInstruction decoded;
    uint32_t warp_id;
    uint32_t pc;
};

class InstructionBuffer {
public:
    explicit InstructionBuffer(uint32_t depth) : max_depth_(depth) {}

    bool is_full() const { return entries_.size() >= max_depth_; }
    bool is_empty() const { return entries_.empty(); }
    uint32_t size() const { return static_cast<uint32_t>(entries_.size()); }

    void push(const BufferEntry& entry) {
        if (!is_full()) {
            entries_.push_back(entry);
        }
    }

    const BufferEntry& front() const { return entries_.front(); }
    BufferEntry& front() { return entries_.front(); }

    void pop() {
        if (!entries_.empty()) {
            entries_.pop_front();
        }
    }

    void flush() { entries_.clear(); }

    void reset() { entries_.clear(); }

private:
    uint32_t max_depth_;
    std::deque<BufferEntry> entries_;
};

} // namespace gpu_sim
