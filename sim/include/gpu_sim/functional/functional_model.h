#pragma once

#include "gpu_sim/config.h"
#include "gpu_sim/trace_event.h"
#include "gpu_sim/functional/register_file.h"
#include "gpu_sim/functional/memory.h"

namespace gpu_sim {

class FunctionalModel {
public:
    explicit FunctionalModel(const SimConfig& config);

    // Execute the instruction at the given PC for the given warp.
    // Updates register file and memory state. Returns a TraceEvent.
    TraceEvent execute(WarpId warp_id, uint32_t pc);

    // State accessors
    FunctionalRegisterFile& register_file() { return reg_file_; }
    const FunctionalRegisterFile& register_file() const { return reg_file_; }
    FlatMemory& memory() { return memory_; }
    const FlatMemory& memory() const { return memory_; }
    InstructionMemory& instruction_memory() { return instr_mem_; }
    const InstructionMemory& instruction_memory() const { return instr_mem_; }
    LookupTable& lookup_table() { return lookup_table_; }
    const LookupTable& lookup_table() const { return lookup_table_; }

    // Warp state
    bool is_warp_active(WarpId warp) const { return warp_active_[warp]; }
    void set_warp_active(WarpId warp, bool active) { warp_active_[warp] = active; }
    bool is_panicked() const { return panicked_; }
    uint32_t panic_warp() const { return panic_warp_; }
    uint32_t panic_cause() const { return panic_cause_; }
    uint32_t panic_pc() const { return panic_pc_; }

    // Initialize all warps for kernel launch
    void init_kernel(const SimConfig& config);

    // Reset all state
    void reset();

private:
    uint32_t execute_load(MemOp op, uint32_t addr);
    void execute_store(MemOp op, uint32_t addr, uint32_t data);
    uint8_t mem_access_size(MemOp op) const;

    SimConfig config_;
    FunctionalRegisterFile reg_file_;
    FlatMemory memory_;
    InstructionMemory instr_mem_;
    LookupTable lookup_table_;

    bool warp_active_[MAX_WARPS]{};
    bool panicked_ = false;
    uint32_t panic_warp_ = 0;
    uint32_t panic_cause_ = 0;
    uint32_t panic_pc_ = 0;

    uint64_t sequence_counter_ = 0;
};

} // namespace gpu_sim
