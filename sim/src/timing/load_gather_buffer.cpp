#include "gpu_sim/timing/load_gather_buffer.h"
#include <cassert>

namespace gpu_sim {

LoadGatherBufferFile::LoadGatherBufferFile(uint32_t num_warps, Stats& stats)
    : num_warps_(num_warps), stats_(stats) {
    // Phase 5b: enroll every Reg so commit_all() / reset_all() drive them
    // uniformly. The constructor sizes the buffers_ vector to num_warps_
    // entries (Reg's default value-initialization leaves it empty); set_next
    // + commit() leaves both current_ and next_ sized (commit copies
    // current_=next_). Reg::initialize() sizes both slots in one step.
    register_state(&buffers_, &rr_pointer_, &claim_request_, &has_result_);
    buffers_.initialize(std::vector<LoadGatherBuffer>(num_warps));
}

bool LoadGatherBufferFile::current_busy(uint32_t warp_id) const {
    // Coalescing's per-warp claim gate (memory plan M2). evaluate() writes the
    // freshly-applied claim's busy flag into buffers_.current_mut() as well as
    // buffers_.next_mut(), so a claim applied earlier in this same tick is
    // visible here even though the commit-phase flip has not run yet.
    return buffers_.current()[warp_id].busy;
}

void LoadGatherBufferFile::claim(uint32_t warp_id, uint8_t dest_reg, uint32_t pc,
                                 uint64_t issue_cycle, uint32_t raw_instruction) {
    // Phase M2: REGISTERED claim. Writes only into the staged claim_request_;
    // gather_file_.evaluate() at the next tick reads the committed
    // claim_request_ (after commit flips next -> current) and applies the
    // metadata + busy mutation. Coalescing's per-warp gate
    // (current_busy(warp)) rejects double claims because a stale
    // buf.busy=false read at cycle N+1 cannot race with an unapplied claim —
    // gather_file.evaluate runs before coalescing.evaluate in the tick sweep.
    auto& req = claim_request_.next_mut();
    assert(!req.valid && "claim() with a pending claim slot");
    req.valid = true;
    req.warp_id = warp_id;
    req.dest_reg = dest_reg;
    req.pc = pc;
    req.issue_cycle = issue_cycle;
    req.raw_instruction = raw_instruction;
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
    // bail out. `commit()` resets the wire at end of cycle.
    // The port is a single shared resource (spec §5.3 Port model: one
    // line-to-gather-buffer extraction per cycle, FILL > secondary > HIT).
    // Phase 7 (reg.h migration): backed by Wire<bool>.
    if (next_port_claimed_.value()) {
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

    // Phase 10D: the fill lands in the staged buffers. seed_next() at the top
    // of the tick seeded next from current, so the accumulation below extends
    // committed state; commit() flips next -> current.
    auto& buf = buffers_.next_mut()[warp_id];
    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        if ((lane_mask >> i) & 1u) {
            if (!buf.slot_valid[i]) {
                buf.filled_count++;
            }
            buf.values[i] = values[i];
            buf.slot_valid[i] = true;
        }
    }
    next_port_claimed_.drive(true);
    return true;
}

void LoadGatherBufferFile::seed_next() {
    // Phase 10D + 5b: seed the carry-forward Regs so try_write() accumulates
    // onto committed state and unwritten buffers carry through unchanged.
    // The gather buffer is NOT frozen by the writeback stall, so this runs
    // unconditionally every cycle.
    //
    // Phase 5b note: claim_request_ is deliberately NOT seeded (auto-seed
    // would re-latch a consumed claim into the staged slot — the same
    // memoryless-consumer rationale as L1Cache's load_cmd_/store_cmd_ in
    // Phase 5a). buffers_, rr_pointer_, and has_result_ DO seed: buffers_
    // matches the pre-Phase-5b explicit copy; rr_pointer_ and has_result_
    // are byte-identical because today's code already held their value
    // across cycles (rr_pointer_ persists as a plain uint, current_has_result_
    // is overwritten at the end of commit so the seed is also overwritten).
    buffers_.seed();
    rr_pointer_.seed();
    has_result_.seed();
}

void LoadGatherBufferFile::evaluate() {
    // Phase M2: apply a deferred claim if one is pending. The claim mutation
    // sets only metadata + busy. Same-cycle write ordering: this evaluate
    // runs before cache.evaluate() in tick(), so any FILL/secondary write
    // deposited this cycle observes the freshly-applied claim metadata.
    //
    // Phase 10D: the claim is written into BOTH the committed buffers
    // (buffers_.current_mut()) and the staged buffers (buffers_.next_mut()).
    // The committed write makes the busy flag visible to coalescing's
    // same-cycle current_busy() gate (memory plan M2); the staged write
    // carries the claim across the commit-phase flip. seed_next() ran
    // earlier this tick (next == old current), so both copies start
    // consistent; writing both keeps them consistent through commit(). A
    // fresh claim lands on an idle buffer, so the per-lane fill fields are
    // already cleared (the prior consume_result/release reset them) and need
    // no touch here.
    //
    // Phase 5b: the committed-state write uses the documented
    // Reg::current_mut() escape hatch (a redirect-style mid-cycle override
    // of committed state). The staged-state write is a normal next_mut()
    // staged write.
    if (claim_request_.current().valid) {
        const auto& req = claim_request_.current();
        auto& current_buffers = buffers_.current_mut();
        auto& next_buffers = buffers_.next_mut();
        assert(!current_buffers[req.warp_id].busy &&
               "deferred claim landing on a busy gather buffer");
        for (auto* buffers : {&current_buffers, &next_buffers}) {
            auto& buf = (*buffers)[req.warp_id];
            buf.busy = true;
            buf.dest_reg = req.dest_reg;
            buf.pc = req.pc;
            buf.issue_cycle = req.issue_cycle;
            buf.raw_instruction = req.raw_instruction;
        }
        // Memoryless-consumer mid-cycle invalidation of committed state — the
        // documented Reg::current_mut() escape hatch (Phase 5a precedent in
        // L1Cache: `load_cmd_.current_mut().valid = false`).
        claim_request_.current_mut().valid = false;
    }
}

void LoadGatherBufferFile::commit() {
    // Reset the port-claim wire so the next tick starts with the port free.
    // Phase 7: this stays in commit() (not at the top of evaluate as the
    // cross-stage Wire pattern dictates) — the claim must persist across
    // the whole tick (FILL/secondary/HIT all run within one tick), only
    // clearing at the cycle boundary. Preserves pre-Phase-7 timing exactly.
    next_port_claimed_.reset();

    // Phase 10D: apply the REGISTERED buffer release staged by
    // consume_result() this cycle. This is the commit-phase effect that
    // replaces consume_result()'s former in-place mutation — busy/slot_valid/
    // filled_count are reset in the staged buffer copy (the about-to-be-
    // committed value), and the round-robin pointer advances past the retired
    // buffer.
    if (next_release_.valid) {
        auto& buf = buffers_.next_mut()[next_release_.warp_id];
        buf.busy = false;
        buf.slot_valid.fill(false);
        buf.filled_count = 0;
        rr_pointer_.set_next((next_release_.warp_id + 1) % num_warps_);
        next_release_ = GatherReleaseRequest{};
    }

    // Phase M4 + 5b: recompute the REGISTERED has-result flag from the
    // about-to-be-committed buffer state (i.e. buffers_.next(), the value
    // that commit_all() below is about to latch into buffers_.current()).
    // Stage it via set_next so commit_all() flips it atomically with
    // buffers_. Scanning is O(num_warps), bounded.
    bool any_full = false;
    for (const auto& buf : buffers_.next()) {
        if (buf.busy && buf.filled_count == WARP_SIZE) {
            any_full = true;
            break;
        }
    }
    has_result_.set_next(any_full);

    // Phase 5b: flip every enrolled Reg in one sweep (buffers_, rr_pointer_,
    // claim_request_, has_result_). After this, buffers_.current() reflects
    // all of this cycle's try_write() fills, the freshly-applied claim, and
    // any release; rr_pointer_, has_result_, and claim_request_ are
    // similarly latched. Byte-identical to the pre-Phase-5b ordering: the
    // hand-rolled code applied the release first (mutating both copies of
    // buffers_), then `current_buffers_ = next_buffers_`, then
    // `current_claim_request_ = next_claim_request_`. The release mutation
    // above touches only buffers_.next_mut() and rr_pointer_.next_mut(); the
    // commit_all() flip then latches current = next for both. This is
    // observationally equivalent because no reader runs between the release
    // mutation and the flip.
    commit_all();

    // Phase M2 + 5b: clear the staged claim_request_. Today's hand-rolled
    // commit cleared `next_claim_request_ = GatherClaimRequest{}` AFTER
    // copying to current_; the Reg-flipped equivalent is set_next(empty)
    // after commit_all has flipped current = next. Without this re-clear, a
    // subsequent commit() with no intervening claim() call would re-latch
    // the same claim into current — the memoryless-consumer contract is
    // identical to L1Cache's load_cmd_ / store_cmd_ in Phase 5a.
    claim_request_.set_next(GatherClaimRequest{});
}

void LoadGatherBufferFile::reset() {
    // Phase 5b: reset_all() value-initializes the Reg's inner storage —
    // buffers_ becomes an empty vector; rr_pointer_, has_result_, and
    // claim_request_ become T{}. Re-size buffers_ back to num_warps_ via the
    // Re-establish the num_warps_ sizing via Reg::initialize. Then clear the
    // plain scratch / staging slots by hand.
    reset_all();
    buffers_.initialize(std::vector<LoadGatherBuffer>(num_warps_));
    // Phase 7: Wire<bool>::reset() de-asserts (default false) — equivalent
    // to the prior `= false` clear.
    next_port_claimed_.reset();
    next_release_ = GatherReleaseRequest{};
}

void LoadGatherBufferFile::flush() {
    reset();
}

WritebackEntry LoadGatherBufferFile::consume_result() {
    // Phase 10D: pure read of committed buffer state. The round-robin scan
    // observes only buffers_.current() — try_write() this cycle lands in
    // buffers_.next_mut() — so the result is independent of where wb_arbiter
    // sits in the evaluate sweep relative to the cache. The buffer is not
    // released here; that committed-state mutation is staged into
    // next_release_ and applied by commit().
    const auto& current_buffers = buffers_.current();
    const uint32_t rr = rr_pointer_.current();
    for (uint32_t i = 0; i < num_warps_; ++i) {
        uint32_t idx = (rr + i) % num_warps_;
        const auto& buf = current_buffers[idx];
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
