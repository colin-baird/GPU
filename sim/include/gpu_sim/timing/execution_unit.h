#pragma once

#include "gpu_sim/types.h"
#include "gpu_sim/trace_event.h"
#include <array>

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

} // namespace gpu_sim
