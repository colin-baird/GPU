#include "gpu_sim/timing/load_gather_buffer.h"
#include <cassert>

namespace gpu_sim {

LoadGatherBufferFile::LoadGatherBufferFile(uint32_t num_warps, Stats& stats)
    : num_warps_(num_warps), stats_(stats), buffers_(num_warps) {}

bool LoadGatherBufferFile::is_busy(uint32_t warp_id) const {
    return buffers_[warp_id].busy;
}

void LoadGatherBufferFile::claim(uint32_t warp_id, uint8_t dest_reg, uint32_t pc,
                                 uint64_t issue_cycle, uint32_t raw_instruction) {
    auto& buf = buffers_[warp_id];
    assert(!buf.busy && "claim() on a busy gather buffer");
    buf.busy = true;
    buf.dest_reg = dest_reg;
    buf.values = {};
    buf.slot_valid.fill(false);
    buf.filled_count = 0;
    buf.pc = pc;
    buf.issue_cycle = issue_cycle;
    buf.raw_instruction = raw_instruction;
    buf.port_used_this_cycle = false;
}

bool LoadGatherBufferFile::try_write(uint32_t warp_id, uint32_t lane_mask,
                                     const std::array<uint32_t, WARP_SIZE>& values,
                                     GatherWriteSource source) {
    auto& buf = buffers_[warp_id];
    if (buf.port_used_this_cycle) {
        // HIT vs FILL port conflict. FILL is scheduled first each cycle, so
        // a false return here is always the HIT path losing to a same-cycle
        // FILL. Two FILLs on one buffer cannot occur because a warp has at
        // most one outstanding load.
        if (source == GatherWriteSource::HIT) {
            stats_.gather_buffer_port_conflict_cycles++;
        }
        return false;
    }

    uint32_t newly_valid = 0;
    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        if ((lane_mask >> i) & 1u) {
            if (!buf.slot_valid[i]) {
                newly_valid++;
            }
            buf.values[i] = values[i];
            buf.slot_valid[i] = true;
        }
    }
    buf.filled_count += newly_valid;
    buf.port_used_this_cycle = true;
    return true;
}

void LoadGatherBufferFile::evaluate() {
    // Work is driven by cache calls into try_write(); nothing to do here.
}

void LoadGatherBufferFile::commit() {
    for (auto& buf : buffers_) {
        buf.port_used_this_cycle = false;
    }
}

void LoadGatherBufferFile::reset() {
    for (auto& buf : buffers_) {
        buf = LoadGatherBuffer{};
    }
    rr_pointer_ = 0;
}

void LoadGatherBufferFile::flush() {
    // Phase 6: panic-flush. Same body as reset() — clear every gather
    // buffer and reset the round-robin pointer.
    for (auto& buf : buffers_) {
        buf = LoadGatherBuffer{};
    }
    rr_pointer_ = 0;
}

bool LoadGatherBufferFile::is_ready() const {
    return true;
}

bool LoadGatherBufferFile::has_result() const {
    for (const auto& buf : buffers_) {
        if (buf.busy && buf.filled_count == WARP_SIZE) {
            return true;
        }
    }
    return false;
}

WritebackEntry LoadGatherBufferFile::consume_result() {
    for (uint32_t i = 0; i < num_warps_; ++i) {
        uint32_t idx = (rr_pointer_ + i) % num_warps_;
        auto& buf = buffers_[idx];
        if (buf.busy && buf.filled_count == WARP_SIZE) {
            WritebackEntry wb;
            wb.valid = true;
            wb.warp_id = idx;
            wb.dest_reg = buf.dest_reg;
            wb.values = buf.values;
            wb.source_unit = ExecUnit::LDST;
            wb.pc = buf.pc;
            wb.raw_instruction = buf.raw_instruction;
            wb.issue_cycle = buf.issue_cycle;

            // Release buffer.
            buf.busy = false;
            buf.slot_valid.fill(false);
            buf.filled_count = 0;

            rr_pointer_ = (idx + 1) % num_warps_;
            return wb;
        }
    }
    // Caller must check has_result() first.
    return WritebackEntry{};
}

} // namespace gpu_sim
