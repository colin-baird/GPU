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
// and rr_pointer_ are Reg<T>. The file also holds two non-register slots —
// next_port_claimed_ (combinational, intra-tick port arbitration; pre-Phase-7
// flavor of cache.next_cmd_ready_) and next_release_ (an evaluate-staging
// slot with no committed twin, applied at commit). Phase 7 will decide
// whether next_port_claimed_ migrates to a Wire<bool>; that is out of scope
// here.
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
    // writer in a tick wins by reading next_port_claimed_, and the others
    // bail. commit() clears the flag at end-of-cycle so the next tick starts
    // unclaimed.
    bool next_port_claimed_ = false;  // combinational, intra-tick port claim (Phase 7 -> Wire<bool>)

    // Phase M2 + 5b: REGISTERED claim-request slot. claim() writes
    // claim_request_.next_mut(); commit() flips next -> current; evaluate()
    // consumes the committed slot at the top of the next tick.
    //
    // Phase 5b note: NOT seeded by seed_next() — auto-seed (next_=current_)
    // would re-latch a consumed claim back into the staged slot on the cycle
    // after evaluate() cleared the committed copy, breaking the
    // memoryless-consumer contract. Same opt-out rationale as L1Cache's
    // load_cmd_ / store_cmd_ in Phase 5a. claim_request_ is still enrolled
    // in register_state() so commit_all() / reset_all() drive it; commit()
    // explicitly clears the staged slot via set_next(GatherClaimRequest{})
    // after commit_all() flips, matching the pre-Phase-5b
    // `next_claim_request_ = GatherClaimRequest{}` tail-of-commit clear.
    Reg<GatherClaimRequest> claim_request_;

    // Phase 10D: REGISTERED buffer-release slot. consume_result() reads the
    // committed buffer (a pure read) and stages a release here; commit()
    // applies it. At most one buffer retires per cycle (single writeback
    // arbiter port), so a single-slot request is sufficient. Phase 5b: kept
    // as a plain staging slot (not a Reg) — it has no committed twin
    // (consume_result writes it, commit consumes-and-clears it within the
    // same tick), so the Reg discipline does not apply.
    GatherReleaseRequest next_release_;  // staging (applied at commit), not a register

    // Phase M4 + 5b: REGISTERED has-result flag. commit() recomputes it from
    // the about-to-be-committed buffer state (true iff any buffer has busy &&
    // filled_count==WARP_SIZE), stages it via set_next, then commit_all()
    // flips. The writeback arbiter reads it via current_has_result() ->
    // has_result_.current().
    Reg<bool> has_result_;
};

} // namespace gpu_sim
