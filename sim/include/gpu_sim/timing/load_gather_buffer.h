#pragma once

#include "gpu_sim/types.h"
#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/timing/reg.h"
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
};

// Phase M2: REGISTERED claim-request slot. Coalescing's claim() writes the
// staged claim_request_; gather_file_.evaluate() consumes the committed
// claim_request_ at the top of the next tick and applies the metadata + busy
// mutation.
struct GatherClaimRequest {
    bool valid = false;
    uint32_t warp_id = 0;
    uint8_t dest_reg = 0;
    uint32_t pc = 0;
    uint64_t issue_cycle = 0;
    uint32_t raw_instruction = 0;
};

// Phase 10D: REGISTERED buffer-release slot. consume_result() is a pure read
// of committed buffer state; instead of mutating the consumed buffer in place
// it stages the release here. commit() applies the release (busy=false,
// slot_valid cleared, filled_count=0) and advances the round-robin pointer.
// Cost: a gather result re-claim by the same warp is visible one cycle later
// (the release lands at commit, not mid-evaluate). This is the sole intended
// cycle delta of Phase 10D.
//
// Phase 1 (reg-family closeout): backed by Wire<GatherReleaseRequest> — see
// next_release_ in LoadGatherBufferFile. The wire is driven inside
// consume_result() (a same-cycle backward-going staging signal) and consumed
// + reset inside commit() at the cycle boundary. Same shape as
// just_claimed_ — intra-class signal whose role spans one tick.
struct GatherReleaseRequest {
    bool valid = false;
    uint32_t warp_id = 0;
};

// One gather buffer per resident warp, registered with the writeback arbiter
// as an ExecutionUnit source so its full-32-lane writebacks arbitrate alongside
// ALU/MUL/DIV/TLOOKUP results.
//
// Phase 10D: the fill presentation is genuinely double-buffered. The cache's
// FILL/secondary/HIT writes land in the staged buffers; commit() flips
// staged -> committed; consume_result() reads committed buffers and is a pure
// read of committed state. This makes the cache -> gather -> wb_arbiter
// forward edge a REGISTERED edge whose result is independent of where
// wb_arbiter sits in the evaluate sweep relative to the cache. The claim's
// busy/metadata mutation (evaluate()) is written into BOTH copies so that
// coalescing's same-cycle current_busy() gate (memory plan M2) still observes
// a freshly-applied claim.
//
// Phase 5b (reg.h migration): all of buffers_, claim_request_, has_result_,
// and rr_pointer_ are Reg<T>. Phase 1 (reg-family closeout): next_release_
// is now Wire<GatherReleaseRequest> — the intra-class staging slot
// consume_result() drives and commit() consumes + resets at the cycle
// boundary, matching the just_claimed_ pattern. next_port_claimed_ remains
// a Wire<bool> (Phase 7: intra-class port arbitration — written and read
// inside the class via try_write()/commit(); reset point is the OWNER'S
// commit() rather than the top of evaluate, preserving the pre-Phase-7
// behavior exactly).
class LoadGatherBufferFile : public ExecutionUnit, public RegisteredStage {
public:
    enum class GatherWriteSource { HIT, FILL };

    LoadGatherBufferFile(uint32_t num_warps, Stats& stats);

    bool current_busy(uint32_t warp_id) const;

    // Phase M2: stages a REGISTERED claim into the staged claim_request_. The
    // buffer's busy flag and metadata are applied at the next tick by
    // evaluate() reading the committed claim_request_. Coalescing's per-warp
    // claim/use protocol guarantees at most one claim per cycle (it
    // processes one FIFO entry at a time), so a single-slot request is
    // sufficient.
    void claim(uint32_t warp_id, uint8_t dest_reg, uint32_t pc,
               uint64_t issue_cycle, uint32_t raw_instruction);

    // Attempts to write `values` for the lanes selected by `lane_mask` into the
    // gather buffer for `warp_id`. Returns false iff another source already
    // used this buffer's port this cycle (HIT path can collide with FILL).
    // Phase 10D: the write lands in the staged buffers; it becomes visible via
    // buffer() / current_has_result() after the next commit().
    bool try_write(uint32_t warp_id, uint32_t lane_mask,
                   const std::array<uint32_t, WARP_SIZE>& values,
                   GatherWriteSource source);

    // ExecutionUnit interface
    // Phase 10D: seed_next() seeds staged buffers from committed buffers so
    // the per-cycle accumulating try_write() calls extend committed state and
    // any buffer not written this cycle carries forward. TimingModel::tick()
    // calls this every cycle (the gather buffer is NOT frozen by the
    // writeback stall). evaluate() applies a deferred claim from the committed
    // claim_request_; the tick sweep places this evaluate before
    // cache.evaluate() so any same-cycle FILL/HIT write observes the
    // freshly-applied claim metadata.
    void seed_next() override;
    void evaluate() override;
    void commit() override;
    void reset() override;

    // Panic flush hook. Called at the commit-phase boundary when the panic
    // signal becomes active. Delegates to reset().
    void flush();
    // Phase M4 / 10B.3: REGISTERED has-result. Returns the committed
    // has_result_ flag (latched at commit when filled_count reaches WARP_SIZE
    // for any buffer); the writeback arbiter and timing-model drain checks see
    // committed state for this source.
    bool current_has_result() const override { return has_result_.current(); }
    // Phase 10D: pure read of committed buffers_; stages a release into
    // next_release_ which commit() applies.
    WritebackEntry consume_result() override;
    ExecUnit get_type() const override { return ExecUnit::LDST; }
    // LoadGatherBufferFile is a writeback source (consumed by the writeback
    // arbiter), never a scheduler dispatch target. The scheduler routes
    // LDST instructions to LdStUnit, so current_busy() is a constant false
    // that the scheduler never queries.
    bool current_busy() const override { return false; }

    uint32_t num_buffers() const { return num_warps_; }
    // Phase 10D: returns the committed buffer state. A try_write() this cycle
    // is visible only after the next commit().
    const LoadGatherBuffer& buffer(uint32_t warp_id) const {
        return buffers_.current()[warp_id];
    }

private:
    uint32_t num_warps_;  // config
    Stats& stats_;        // config (back-pointer)
    // Phase 5b (reg.h migration): the double-buffered fill presentation is now
    // a Reg<std::vector<LoadGatherBuffer>>. buffers_.current() is the
    // committed state read by consume_result() / buffer() / current_busy();
    // buffers_.next_mut() accumulates this cycle's try_write() fills and the
    // staged claim/release. commit_all() flips next -> current.
    Reg<std::vector<LoadGatherBuffer>> buffers_;
    // Phase 5b: rr_pointer_ is the round-robin scan pointer over completed
    // buffers. commit() advances it (writes next_mut()) when a release is
    // applied; commit_all() flips. Auto-seed (next_=current_) at the top of
    // each tick is identical to the pre-Phase-5b "value just carries forward"
    // semantics.
    Reg<uint32_t> rr_pointer_;
    // Single-port arbitration flag for the cache's line-to-gather-buffer
    // extraction port (§5.3 Port model: FILL > secondary > HIT). Writers
    // (cache.handle_responses, cache.drain_secondary_chain_head,
    // cache.process_load HIT path) all funnel through try_write(); the first
    // writer in a tick wins by reading next_port_claimed_.value(), and the
    // others bail. commit() resets it at end-of-cycle so the next tick starts
    // unclaimed. Phase 7: wrapped as Wire<bool>. Intra-class signal — not
    // cross-stage, so no public next_*() accessor; readers (try_write) read
    // wire_.value() directly. Reset point is the OWNER'S commit() (not the
    // top of evaluate as the cross-stage Wire pattern dictates) — this is
    // faithful to the pre-Phase-7 reset timing and necessary because the
    // claim must persist across the whole tick (FILL/secondary/HIT all run
    // within one tick), only clearing at the cycle boundary.
    Wire<bool> next_port_claimed_;

    // Phase 3 of current_mut() elimination (Pattern 4): per-warp "claim
    // applied this cycle" signal. Driven by evaluate() when a deferred claim
    // is applied (the warp_id's bit is set; the rest stay false). Read by
    // current_busy(warp_id), combinationally OR'd with the committed busy
    // flag — synthesis-faithful encoding of "coalescing sees the fresh claim
    // the same cycle." Reset point is the OWNER'S commit() (the same
    // convention as next_port_claimed_ above, not the cross-stage Wire's
    // top-of-evaluate reset) because the wire's role is one full tick:
    // producer (gather) runs at sweep position #2, consumer (coalescing) at
    // #6, both within the same tick, and the wire should de-assert at the
    // cycle boundary. Replaces the previous buffers_.current_mut() dual-write
    // at the claim-apply site.
    Wire<std::array<bool, MAX_WARPS>> just_claimed_;

    // Phase M2 + Phase 4 of current_mut() elimination (Pattern 3):
    // PulseReg<GatherClaimRequest>. claim() writes claim_request_.next_mut();
    // commit() flips next -> current; evaluate() consumes the committed slot
    // at the top of the next tick. seed_next() defaults next_ to T{} each
    // tick via PulseReg::seed(), so a cycle on which coalescing does not
    // claim latches the slot to invalid at commit — the memoryless-consumer
    // contract encoded in the type, replacing the previous mid-cycle
    // current_mut().valid=false clear and tail-of-commit set_next(T{})
    // explicit reset.
    PulseReg<GatherClaimRequest> claim_request_;

    // Phase 10D: REGISTERED buffer-release slot. consume_result() reads the
    // committed buffer (a pure read) and drives the wire here; commit()
    // reads it and applies the release (busy=false, slot_valid cleared,
    // filled_count=0, rr_pointer_ advanced). At most one buffer retires per
    // cycle (single writeback arbiter port), so a single-slot signal is
    // sufficient.
    //
    // Phase 1 (reg-family closeout): Wire<GatherReleaseRequest>. The role
    // is the same as just_claimed_ above: an intra-class signal whose
    // producer (consume_result()) and consumer (commit()) both run within
    // one tick. consume_result() drives the wire; commit() reads .value(),
    // applies the release, then reset()s the wire at the cycle boundary —
    // matching the convention of next_port_claimed_ / just_claimed_ (reset
    // point is the OWNER'S commit()). Not a Reg because it has no committed
    // twin: consume_result() and commit() both run inside the same tick,
    // so the wire only needs to live for the span of one evaluate-commit
    // pair. Not enrolled via register_state() (Wire is not a RegBase).
    Wire<GatherReleaseRequest> next_release_;

    // Phase M4 + 5b: REGISTERED has-result flag. commit() recomputes it from
    // the about-to-be-committed buffer state (true iff any buffer has busy &&
    // filled_count==WARP_SIZE), stages it via set_next, then commit_all()
    // flips. The writeback arbiter reads it via current_has_result() ->
    // has_result_.current().
    Reg<bool> has_result_;
};

} // namespace gpu_sim
