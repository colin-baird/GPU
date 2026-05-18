#include "gpu_sim/timing/load_gather_buffer.h"
#include <cassert>

namespace gpu_sim {

LoadGatherBufferFile::LoadGatherBufferFile(uint32_t num_warps, Stats& stats)
    : num_warps_(num_warps), stats_(stats),
      current_buffers_(num_warps), next_buffers_(num_warps) {}

bool LoadGatherBufferFile::current_busy(uint32_t warp_id) const {
    // Coalescing's per-warp claim gate (memory plan M2). evaluate() writes the
    // freshly-applied claim's busy flag into current_buffers_ as well as
    // next_buffers_, so a claim applied earlier in this same tick is visible
    // here even though the commit-phase flip has not run yet.
    return current_buffers_[warp_id].busy;
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

    // Phase 10D: the fill lands in next_buffers_. seed_next() at the top of
    // the tick copied current_buffers_ -> next_buffers_, so the accumulation
    // below extends committed state; commit() flips next_ -> current_.
    auto& buf = next_buffers_[warp_id];
    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        if ((lane_mask >> i) & 1u) {
            if (!buf.slot_valid[i]) {
                buf.filled_count++;
            }
            buf.values[i] = values[i];
            buf.slot_valid[i] = true;
        }
    }
    next_port_claimed_ = true;
    return true;
}

void LoadGatherBufferFile::seed_next() {
    // Phase 10D: copy committed state forward so try_write() accumulates onto
    // it and unwritten buffers carry through unchanged. The gather buffer is
    // NOT frozen by the writeback stall, so this runs unconditionally every
    // cycle.
    next_buffers_ = current_buffers_;
}

void LoadGatherBufferFile::evaluate() {
    // Phase M2: apply a deferred claim if one is pending. The claim mutation
    // sets only metadata + busy. Same-cycle write ordering: this evaluate
    // runs before cache.evaluate() in tick(), so any FILL/secondary write
    // deposited this cycle observes the freshly-applied claim metadata.
    //
    // Phase 10D: the claim is written into BOTH current_buffers_ and
    // next_buffers_. The current_buffers_ write makes the busy flag visible
    // to coalescing's same-cycle current_busy() gate (memory plan M2); the
    // next_buffers_ write carries the claim across the commit-phase flip.
    // seed_next() ran earlier this tick (next_ == old current_), so both
    // copies start consistent; writing both keeps them consistent through
    // commit(). A fresh claim lands on an idle buffer, so the per-lane fill
    // fields are already cleared (the prior consume_result/release reset
    // them) and need no touch here.
    if (current_claim_request_.valid) {
        const auto& req = current_claim_request_;
        assert(!current_buffers_[req.warp_id].busy &&
               "deferred claim landing on a busy gather buffer");
        for (auto* buffers : {&current_buffers_, &next_buffers_}) {
            auto& buf = (*buffers)[req.warp_id];
            buf.busy = true;
            buf.dest_reg = req.dest_reg;
            buf.pc = req.pc;
            buf.issue_cycle = req.issue_cycle;
            buf.raw_instruction = req.raw_instruction;
        }
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
    // Phase 10D: apply the REGISTERED buffer release staged by
    // consume_result() this cycle. This is the commit-phase effect that
    // replaces consume_result()'s former in-place mutation — busy/slot_valid/
    // filled_count are reset in next_buffers_ (the about-to-be-committed
    // copy), and the round-robin pointer advances past the retired buffer.
    if (next_release_.valid) {
        auto& buf = next_buffers_[next_release_.warp_id];
        buf.busy = false;
        buf.slot_valid.fill(false);
        buf.filled_count = 0;
        rr_pointer_ = (next_release_.warp_id + 1) % num_warps_;
        next_release_ = GatherReleaseRequest{};
    }
    // Phase 10D: flip the double-buffered fill presentation. After this flip
    // current_buffers_ reflects all of this cycle's try_write() fills, the
    // freshly-applied claim, and any release.
    current_buffers_ = next_buffers_;
    // Phase M4: recompute the REGISTERED has-result flag from the committed
    // buffer state. Scanning is O(num_warps), bounded.
    bool any_full = false;
    for (const auto& buf : current_buffers_) {
        if (buf.busy && buf.filled_count == WARP_SIZE) {
            any_full = true;
            break;
        }
    }
    current_has_result_ = any_full;
}

void LoadGatherBufferFile::reset() {
    for (auto& buf : current_buffers_) {
        buf = LoadGatherBuffer{};
    }
    for (auto& buf : next_buffers_) {
        buf = LoadGatherBuffer{};
    }
    rr_pointer_ = 0;
    next_port_claimed_ = false;
    current_claim_request_ = GatherClaimRequest{};
    next_claim_request_ = GatherClaimRequest{};
    next_release_ = GatherReleaseRequest{};
    current_has_result_ = false;
}

void LoadGatherBufferFile::flush() {
    reset();
}

WritebackEntry LoadGatherBufferFile::consume_result() {
    // Phase 10D: pure read of committed buffer state. The round-robin scan
    // observes only current_buffers_ as committed at the end of the previous
    // cycle — try_write() this cycle lands in next_buffers_ — so the result
    // is independent of where wb_arbiter sits in the evaluate sweep relative
    // to the cache. The buffer is not released here; that committed-state
    // mutation is staged into next_release_ and applied by commit().
    for (uint32_t i = 0; i < num_warps_; ++i) {
        uint32_t idx = (rr_pointer_ + i) % num_warps_;
        const auto& buf = current_buffers_[idx];
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

            // Stage the buffer release as a commit-phase effect.
            next_release_.valid = true;
            next_release_.warp_id = idx;
            return wb;
        }
    }
    // Caller must check current_has_result() first.
    return WritebackEntry{};
}

} // namespace gpu_sim
