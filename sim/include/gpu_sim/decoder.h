#pragma once

#include "gpu_sim/trace_event.h"
#include <cstdint>

namespace gpu_sim {

class Decoder {
public:
    static DecodedInstruction decode(uint32_t instruction);
};

} // namespace gpu_sim
