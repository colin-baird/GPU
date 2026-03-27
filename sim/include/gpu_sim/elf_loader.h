#pragma once

#include "gpu_sim/functional/functional_model.h"
#include <cstdint>
#include <string>
#include <vector>

namespace gpu_sim {

// A loadable segment from an ELF or raw binary.
struct ProgramSegment {
    uint32_t vaddr;
    std::vector<uint8_t> data;
};

// Model-independent representation of a parsed program binary.
struct ProgramImage {
    uint32_t entry_pc = 0;
    std::vector<ProgramSegment> segments;
};

// Parse a program binary into a ProgramImage without loading into any model.
// Supports raw binary (flat sequence of 32-bit LE instruction words)
// and ELF32 RISC-V binaries.
ProgramImage load_program_image(const std::string& path);

// Load a program binary into the functional model's instruction memory.
// Convenience wrapper: calls load_program_image() then copies segments
// into the model's memories.
// Returns the entry point PC (0 for raw binary, e_entry for ELF).
uint32_t load_program(FunctionalModel& model, const std::string& path);

// Load a ProgramImage into a functional model's memories.
// Segments with vaddr below instruction_mem_size go to instruction memory,
// others go to data memory.
void load_image_into_model(FunctionalModel& model, const ProgramImage& image);

// Load a lookup table from a raw binary file (sequence of 32-bit LE values).
void load_lookup_table(FunctionalModel& model, const std::string& path);

// Load raw data into external memory at a given address.
void load_data(FunctionalModel& model, const std::string& path, uint32_t base_addr);

} // namespace gpu_sim
