#pragma once

#include "gpu_sim/types.h"
#include "gpu_sim/trace_event.h"
#include <array>
#include <deque>

namespace gpu_sim {

struct WritebackEntry {
    bool valid = false;
    uint32_t warp_id = 0;
    uint8_t dest_reg = 0;
    std::array<uint32_t, WARP_SIZE> values{};
    ExecUnit source_unit = ExecUnit::NONE;
    uint32_t pc = 0;
    uint32_t raw_instruction = 0;
    uint64_t issue_cycle = 0;  // For latency tracking
};

class ExecutionUnit {
public:
    virtual ~ExecutionUnit() = default;
    virtual void evaluate() = 0;
    virtual void commit() = 0;
    virtual void reset() = 0;
    virtual bool is_ready() const = 0;
    virtual bool has_result() const = 0;
    virtual WritebackEntry consume_result() = 0;
    virtual ExecUnit get_type() const = 0;
};

class QueuedWritebackSource : public ExecutionUnit {
public:
    explicit QueuedWritebackSource(ExecUnit type) : type_(type) {}

    void enqueue(const WritebackEntry& entry) {
        if (entry.valid) {
            queue_.push_back(entry);
        }
    }

    void evaluate() override {}
    void commit() override {}
    void reset() override { queue_.clear(); }
    bool is_ready() const override { return true; }
    bool has_result() const override { return !queue_.empty(); }

    WritebackEntry consume_result() override {
        WritebackEntry entry = queue_.front();
        queue_.pop_front();
        return entry;
    }

    ExecUnit get_type() const override { return type_; }

    const WritebackEntry* front_entry() const {
        return queue_.empty() ? nullptr : &queue_.front();
    }

    size_t queue_depth() const { return queue_.size(); }

private:
    ExecUnit type_;
    std::deque<WritebackEntry> queue_;
};

} // namespace gpu_sim
