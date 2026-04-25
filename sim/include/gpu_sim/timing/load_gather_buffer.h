#pragma once

#include "gpu_sim/types.h"
#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/stats.h"
#include <array>
#include <cstdint>
#include <vector>

namespace gpu_sim {

// Per-resident-warp load gather buffer. Holds the per-lane values for an
// in-flight load; the single writeback is emitted once all 32 slots are filled.
// See §5.2.1 of the architectural spec.
struct LoadGatherBuffer {
    bool busy = false;
    uint8_t dest_reg = 0;
    std::array<uint32_t, WARP_SIZE> values{};
    std::array<bool, WARP_SIZE> slot_valid{};
    uint32_t filled_count = 0;
    uint32_t pc = 0;
    uint64_t issue_cycle = 0;
    uint32_t raw_instruction = 0;
    // Scratch: cleared every commit(); limits buffer writes to one per cycle.
    bool port_used_this_cycle = false;
};

// One gather buffer per resident warp, registered with the writeback arbiter
// as an ExecutionUnit source so its full-32-lane writebacks arbitrate alongside
// ALU/MUL/DIV/TLOOKUP results.
class LoadGatherBufferFile : public ExecutionUnit {
public:
    enum class GatherWriteSource { HIT, FILL };

    LoadGatherBufferFile(uint32_t num_warps, Stats& stats);

    bool is_busy(uint32_t warp_id) const;

    void claim(uint32_t warp_id, uint8_t dest_reg, uint32_t pc,
               uint64_t issue_cycle, uint32_t raw_instruction);

    // Attempts to write `values` for the lanes selected by `lane_mask` into the
    // gather buffer for `warp_id`. Returns false iff another source already
    // used this buffer's port this cycle (HIT path can collide with FILL).
    bool try_write(uint32_t warp_id, uint32_t lane_mask,
                   const std::array<uint32_t, WARP_SIZE>& values,
                   GatherWriteSource source);

    // ExecutionUnit interface
    void evaluate() override;
    void commit() override;
    void reset() override;
    bool is_ready() const override;
    bool has_result() const override;
    WritebackEntry consume_result() override;
    ExecUnit get_type() const override { return ExecUnit::LDST; }
    // Phase 4: LoadGatherBufferFile is a writeback source (consumed by the
    // writeback arbiter), never a scheduler dispatch target. The scheduler
    // routes LDST instructions to LdStUnit. Default no-op compute_ready()
    // suffices; ready_out() returns true since the scheduler never queries it.
    bool ready_out() const override { return true; }

    uint32_t num_buffers() const { return num_warps_; }
    const LoadGatherBuffer& buffer(uint32_t warp_id) const { return buffers_[warp_id]; }

private:
    uint32_t num_warps_;
    Stats& stats_;
    std::vector<LoadGatherBuffer> buffers_;
    uint32_t rr_pointer_ = 0;
};

} // namespace gpu_sim
