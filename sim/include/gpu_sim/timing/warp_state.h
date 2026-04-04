#pragma once

#include "gpu_sim/types.h"
#include "gpu_sim/timing/instruction_buffer.h"

namespace gpu_sim {

struct WarpState {
    uint32_t pc = 0;
    bool active = false;
    bool branch_in_flight = false;
    InstructionBuffer instr_buffer;

    explicit WarpState(uint32_t buffer_depth = 3) : instr_buffer(buffer_depth) {}

    void reset(uint32_t start_pc) {
        pc = start_pc;
        active = true;
        branch_in_flight = false;
        instr_buffer.reset();
    }
};

} // namespace gpu_sim
