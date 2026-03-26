#pragma once

#include "gpu_sim/types.h"
#include <cstdint>

namespace gpu_sim {

// Pure functions: given inputs, produce outputs. No state.
uint32_t execute_alu(AluOp op, uint32_t a, uint32_t b);
uint32_t execute_mul(MulDivOp op, uint32_t a, uint32_t b);
uint32_t execute_div(MulDivOp op, uint32_t a, uint32_t b);
uint32_t execute_vdot8(uint32_t rs1, uint32_t rs2, uint32_t rd_accum);

// Branch evaluation: returns true if branch is taken
bool evaluate_branch(BranchOp op, uint32_t a, uint32_t b);

} // namespace gpu_sim
