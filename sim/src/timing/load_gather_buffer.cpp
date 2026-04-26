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
    // Phase 7: claim() does NOT touch the shared port-claim flag —
    // arbitration is owned at the LoadGatherBufferFile level (single
    // physical extraction port per spec §5.3). A fresh claim does not
    // imply a port write; the subsequent try_write does.
}

bool LoadGatherBufferFile::try_write(uint32_t warp_id, uint32_t lane_mask,
                                     const std::array<uint32_t, WARP_SIZE>& values,
                                     GatherWriteSource source) {
    // Phase 7: read the live `next_port_claimed_` flag — within a single
    // tick, multiple try_write() calls happen sequentially (FILL first via
    // cache.handle_responses, then secondary via drain_secondary_chain_head,
    // then HIT via coalescing.evaluate -> cache.process_load). The first
    // writer wins; subsequent same-tick writers see the updated `next_*` and
    // bail out. `commit()` moves next -> current at the end of the cycle.
    // The port is a single shared resource (spec §5.3 Port model: one
    // line-to-gather-buffer extraction per cycle, FILL > secondary > HIT).
    if (next_port_claimed_) {
        // FILL > secondary > HIT priority; a false return is always either
        // the HIT path losing to an earlier FILL/secondary in the same cycle,
        // or a secondary losing to a prior FILL. Two FILLs in one cycle
        // cannot occur because the cache accepts at most one fill response
        // per cycle.
        if (source == GatherWriteSource::HIT) {
            stats_.gather_buffer_port_conflict_cycles++;
        }
        return false;
    }

    auto& buf = buffers_[warp_id];
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
    next_port_claimed_ = true;
    return true;
}

void LoadGatherBufferFile::evaluate() {
    // Work is driven by cache calls into try_write(); nothing to do here.
}

void LoadGatherBufferFile::commit() {
    // Phase 7: REGISTERED single-port flip. Move next -> current and clear
    // next so the upcoming cycle starts with the port free.
    current_port_claimed_ = next_port_claimed_;
    next_port_claimed_ = false;
}

void LoadGatherBufferFile::reset() {
    for (auto& buf : buffers_) {
        buf = LoadGatherBuffer{};
    }
    rr_pointer_ = 0;
    next_port_claimed_ = false;
    current_port_claimed_ = false;
}

void LoadGatherBufferFile::flush() {
    // Phase 6: panic-flush. Same body as reset() — clear every gather
    // buffer, reset the round-robin pointer, and clear the shared port
    // arbitration flags.
    for (auto& buf : buffers_) {
        buf = LoadGatherBuffer{};
    }
    rr_pointer_ = 0;
    next_port_claimed_ = false;
    current_port_claimed_ = false;
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
