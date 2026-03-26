#pragma once

#include "gpu_sim/functional/functional_model.h"
#include <string>

namespace gpu_sim {

// Load a program binary into the functional model's instruction memory.
// Supports raw binary (flat sequence of 32-bit LE instruction words)
// and ELF32 RISC-V binaries.
// Returns the entry point PC (0 for raw binary, e_entry for ELF).
uint32_t load_program(FunctionalModel& model, const std::string& path);

// Load a lookup table from a raw binary file (sequence of 32-bit LE values).
void load_lookup_table(FunctionalModel& model, const std::string& path);

// Load raw data into external memory at a given address.
void load_data(FunctionalModel& model, const std::string& path, uint32_t base_addr);

} // namespace gpu_sim
