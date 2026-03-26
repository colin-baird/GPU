#pragma once

#include "gpu_sim/types.h"
#include <array>
#include <cstdint>
#include <cstring>

namespace gpu_sim {

class FunctionalRegisterFile {
public:
    FunctionalRegisterFile() { reset(); }

    void reset() {
        std::memset(regs_, 0, sizeof(regs_));
    }

    uint32_t read(WarpId warp, LaneId lane, RegIndex reg) const {
        if (reg == 0) return 0;  // r0 hardwired to 0
        return regs_[warp][lane][reg];
    }

    void write(WarpId warp, LaneId lane, RegIndex reg, uint32_t value) {
        if (reg == 0) return;  // Writes to r0 discarded
        regs_[warp][lane][reg] = value;
    }

    // Initialize kernel arguments for all lanes in a warp
    void init_warp(WarpId warp, const uint32_t kernel_args[4]) {
        for (LaneId lane = 0; lane < WARP_SIZE; ++lane) {
            for (RegIndex r = 0; r < NUM_REGS; ++r) {
                regs_[warp][lane][r] = 0;
            }
            regs_[warp][lane][1] = kernel_args[0];
            regs_[warp][lane][2] = kernel_args[1];
            regs_[warp][lane][3] = kernel_args[2];
            regs_[warp][lane][4] = kernel_args[3];
        }
    }

private:
    uint32_t regs_[MAX_WARPS][WARP_SIZE][NUM_REGS];
};

} // namespace gpu_sim
