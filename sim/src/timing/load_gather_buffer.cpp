#include "gpu_sim/timing/load_gather_buffer.h"
#include <cassert>

namespace gpu_sim {

LoadGatherBufferFile::LoadGatherBufferFile(uint32_t num_warps, Stats& stats)
    : num_warps_(num_warps), stats_(stats), buffers_(num_warps) {}

bool LoadGatherBufferFile::current_busy(uint32_t warp_id) const {
    return buffers_[warp_id].busy;
}

void LoadGatherBufferFile::claim(uint32_t warp_id, uint8_t dest_reg, uint32_t pc,
                                 uint64_t issue_cycle, uint32_t raw_instruction) {
    // Phase M2: REGISTERED claim. Writes only into next_claim_request_;
    // gather_file_.evaluate() at the next tick reads current_claim_request_
    // (after commit flips next -> current) and applies the metadata + busy
    // mutation. Coalescing's per-warp gate (current_busy(warp)) rejects
    // double claims because a stale buf.busy=false read at cycle N+1 cannot
    // race with an unapplied claim — gather_file.evaluate runs before
    // coalescing.evaluate in the tick sweep.
    assert(!next_claim_request_.valid && "claim() with a pending claim slot");
    next_claim_request_.valid = true;
    next_claim_request_.warp_id = warp_id;
    next_claim_request_.dest_reg = dest_reg;
    next_claim_request_.pc = pc;
    next_claim_request_.issue_cycle = issue_cycle;
    next_claim_request_.raw_instruction = raw_instruction;
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
    // Phase M4: stage the REGISTERED has-result flag. If this write
    // completed a buffer (busy && filled_count == WARP_SIZE), the buffer
    // is ready for writeback; the arbiter will see it via
    // current_has_result() at the next cycle (commit flips next_ → current_).
    if (buf.busy && buf.filled_count == WARP_SIZE) {
        next_has_result_ = true;
    }
    return true;
}

void LoadGatherBufferFile::evaluate() {
    // Phase M2: apply a deferred claim if one is pending. The claim mutation
    // sets only metadata + busy; consume_result() handles cleanup of
    // slot_valid/filled_count between consecutive uses, so values[] from a
    // prior use are correctly masked by slot_valid=false until try_write
    // overwrites them. Same-cycle write ordering: this evaluate runs before
    // cache.evaluate() in tick(), so any FILL/secondary write deposited
    // this cycle observes the freshly-applied claim metadata.
    if (current_claim_request_.valid) {
        auto& buf = buffers_[current_claim_request_.warp_id];
        assert(!buf.busy && "deferred claim landing on a busy gather buffer");
        buf.busy = true;
        buf.dest_reg = current_claim_request_.dest_reg;
        buf.pc = current_claim_request_.pc;
        buf.issue_cycle = current_claim_request_.issue_cycle;
        buf.raw_instruction = current_claim_request_.raw_instruction;
        current_claim_request_.valid = false;
    }
}

void LoadGatherBufferFile::commit() {
    // Clear the port-claim flag so the next tick starts with the port free.
    next_port_claimed_ = false;
    // Phase M2: flip the REGISTERED claim-request slot. evaluate() at the
    // top of the next tick consumes current_claim_request_.
    current_claim_request_ = next_claim_request_;
    next_claim_request_ = GatherClaimRequest{};
    // Phase M4: flip the REGISTERED has-result flag. Recompute next_ from
    // the current buffer state — try_write may have just produced a fresh
    // full-buffer this cycle (next_has_result_ already true), or
    // consume_result may have just released a buffer (next_has_result_
    // could still be true if another buffer is full). Scanning is O(num_warps),
    // bounded.
    bool any_full = false;
    for (const auto& buf : buffers_) {
        if (buf.busy && buf.filled_count == WARP_SIZE) {
            any_full = true;
            break;
        }
    }
    current_has_result_ = any_full;
    next_has_result_ = any_full;
}

void LoadGatherBufferFile::reset() {
    for (auto& buf : buffers_) {
        buf = LoadGatherBuffer{};
    }
    rr_pointer_ = 0;
    next_port_claimed_ = false;
    current_claim_request_ = GatherClaimRequest{};
    next_claim_request_ = GatherClaimRequest{};
    current_has_result_ = false;
    next_has_result_ = false;
}

void LoadGatherBufferFile::flush() {
    reset();
}

bool LoadGatherBufferFile::next_has_result() const {
    // Phase M4: REGISTERED — return committed flag. The base interface name
    // is preserved for compatibility with other ExecutionUnit overrides;
    // see current_has_result() for the canonical name.
    return current_has_result_;
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
    // Caller must check next_has_result() first.
    return WritebackEntry{};
}

} // namespace gpu_sim
