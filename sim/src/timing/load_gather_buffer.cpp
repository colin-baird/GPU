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
    // Coalescing's per-warp claim gate (memory plan M2). Combines the
    // committed busy flag (latched at last commit) with this cycle's
    // just_claimed_ Wire — driven by evaluate() when a deferred claim is
    // applied earlier in this tick (gather.evaluate runs at sweep position
    // #2, coalescing.evaluate at #6, so the wire reflects this tick's
    // claim by the time coalescing reads). Synthesis-faithful combinational
    // forwarding; previously a buffers_.current_mut() dual-write achieved
    // the same observation by mutating Q mid-cycle.
    return buffers_.current()[warp_id].busy || just_claimed_.value()[warp_id];
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
    // Phase 4 of current_mut() elimination: seed_all() seeds every enrolled
    // Reg / PulseReg. The gather buffer is NOT frozen by the writeback
    // stall, so this runs unconditionally every cycle.
    //
    // buffers_, rr_pointer_, has_result_ are Reg<T>: seed copies next_ =
    // current_ — byte-identical to the pre-Phase-4 explicit copy.
    // claim_request_ is PulseReg<T>: seed defaults next_ = T{} — replaces
    // the previous tail-of-commit set_next(T{}) clear.
    seed_all();
}

void LoadGatherBufferFile::evaluate() {
    // Phase M2: apply a deferred claim if one is pending. The claim mutation
    // sets only metadata + busy. Same-cycle write ordering: this evaluate
    // runs before cache.evaluate() in tick(), so any FILL/secondary write
    // deposited this cycle observes the freshly-applied claim metadata.
    //
    // Phase 3 of current_mut() elimination: the claim writes ONLY into the
    // staged buffers (buffers_.next_mut()); commit() latches. Coalescing's
    // same-cycle current_busy() gate observes the fresh claim via the
    // just_claimed_ Wire — synthesis-faithful combinational forwarding,
    // replacing the previous dual-write to both committed and staged buffer
    // copies. A fresh claim lands on an idle buffer, so the per-lane fill
    // fields are already cleared (the prior consume_result/release reset
    // them) and need no touch here.
    if (claim_request_.current().valid) {
        const auto& req = claim_request_.current();
        assert(!buffers_.current()[req.warp_id].busy &&
               "deferred claim landing on a busy gather buffer");

        auto& buf = buffers_.next_mut()[req.warp_id];
        buf.busy = true;
        buf.dest_reg = req.dest_reg;
        buf.pc = req.pc;
        buf.issue_cycle = req.issue_cycle;
        buf.raw_instruction = req.raw_instruction;

        // Drive the per-warp "claim applied this cycle" bit. Coalescing's
        // current_busy(warp_id) reads (committed[warp].busy || wire[warp]).
        std::array<bool, MAX_WARPS> bits{};
        bits[req.warp_id] = true;
        just_claimed_.drive(bits);
        // Phase 4: claim_request_ is PulseReg<T>. No mid-cycle current_mut()
        // clear; seed_next() at the top of the next tick defaults next_ to
        // T{}, replacing the previous explicit invalidation.
    }
}

void LoadGatherBufferFile::commit() {
    // Reset the port-claim wire so the next tick starts with the port free.
    // Phase 7: this stays in commit() (not at the top of evaluate as the
    // cross-stage Wire pattern dictates) — the claim must persist across
    // the whole tick (FILL/secondary/HIT all run within one tick), only
    // clearing at the cycle boundary. Preserves pre-Phase-7 timing exactly.
    next_port_claimed_.reset();
    // Phase 3 of current_mut() elimination: reset just_claimed_ at the cycle
    // boundary. The wire encodes "claim applied THIS cycle"; by end of tick,
    // gather.evaluate and coalescing.evaluate have both run (positions #2 and
    // #6 in the sweep), so the wire's role is done. Reset here (not at top
    // of evaluate) so a reader that calls current_busy() between commit() and
    // the next evaluate() — for example a test asserting the post-release
    // committed state — observes the cycle as ended.
    just_claimed_.reset();

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

    // Phase 4 of current_mut() elimination: claim_request_ is PulseReg<T>;
    // seed_next() at the top of the next tick defaults next_ to T{},
    // replacing the previous explicit tail set_next(T{}) clear.
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
    just_claimed_.reset();
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
