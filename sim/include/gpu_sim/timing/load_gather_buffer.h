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
};

// Phase M2: REGISTERED claim-request slot. Coalescing's claim() writes
// next_claim_request_; gather_file_.evaluate() consumes current_claim_request_
// at the top of the next tick and applies the metadata + busy mutation.
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
// FILL/secondary/HIT writes land in next_buffers_; commit() flips next_ ->
// current_; consume_result() reads current_buffers_ and is a pure read of
// committed state. This makes the cache -> gather -> wb_arbiter forward edge a
// REGISTERED edge whose result is independent of where wb_arbiter sits in the
// evaluate sweep relative to the cache. The claim's busy/metadata mutation
// (evaluate()) is written into BOTH copies so that coalescing's same-cycle
// current_busy() gate (memory plan M2) still observes a freshly-applied claim.
class LoadGatherBufferFile : public ExecutionUnit {
public:
    enum class GatherWriteSource { HIT, FILL };

    LoadGatherBufferFile(uint32_t num_warps, Stats& stats);

    bool current_busy(uint32_t warp_id) const;

    // Phase M2: stages a REGISTERED claim into next_claim_request_. The
    // buffer's busy flag and metadata are applied at the next tick by
    // evaluate() reading current_claim_request_. Coalescing's per-warp
    // claim/use protocol guarantees at most one claim per cycle (it
    // processes one FIFO entry at a time), so a single-slot request is
    // sufficient.
    void claim(uint32_t warp_id, uint8_t dest_reg, uint32_t pc,
               uint64_t issue_cycle, uint32_t raw_instruction);

    // Attempts to write `values` for the lanes selected by `lane_mask` into the
    // gather buffer for `warp_id`. Returns false iff another source already
    // used this buffer's port this cycle (HIT path can collide with FILL).
    // Phase 10D: the write lands in next_buffers_; it becomes visible via
    // buffer() / current_has_result() after the next commit().
    bool try_write(uint32_t warp_id, uint32_t lane_mask,
                   const std::array<uint32_t, WARP_SIZE>& values,
                   GatherWriteSource source);

    // ExecutionUnit interface
    // Phase 10D: seed_next() copies current_buffers_ -> next_buffers_ so the
    // per-cycle accumulating try_write() calls extend committed state and any
    // buffer not written this cycle carries forward. TimingModel::tick() calls
    // this every cycle (the gather buffer is NOT frozen by the writeback
    // stall). evaluate() applies a deferred claim from current_claim_request_;
    // the tick sweep places this evaluate before cache.evaluate() so any
    // same-cycle FILL/HIT write observes the freshly-applied claim metadata.
    void seed_next() override;
    void evaluate() override;
    void commit() override;
    void reset() override;

    // Panic flush hook. Called at the commit-phase boundary when the panic
    // signal becomes active. Delegates to reset().
    void flush();
    // Phase M4 / 10B.3: REGISTERED has-result. Returns the committed
    // current_has_result_ flag (latched at commit when filled_count reaches
    // WARP_SIZE for any buffer); the writeback arbiter and timing-model drain
    // checks see committed state for this source.
    bool current_has_result() const override { return current_has_result_; }
    // Phase 10D: pure read of committed current_buffers_; stages a release
    // into next_release_ which commit() applies.
    WritebackEntry consume_result() override;
    ExecUnit get_type() const override { return ExecUnit::LDST; }
    // LoadGatherBufferFile is a writeback source (consumed by the writeback
    // arbiter), never a scheduler dispatch target. The scheduler routes
    // LDST instructions to LdStUnit, so current_busy() is a constant false
    // that the scheduler never queries.
    bool current_busy() const override { return false; }

    uint32_t num_buffers() const { return num_warps_; }
    // Phase 10D: returns the committed buffer state (current_buffers_). A
    // try_write() this cycle is visible only after the next commit().
    const LoadGatherBuffer& buffer(uint32_t warp_id) const {
        return current_buffers_[warp_id];
    }

private:
    uint32_t num_warps_;
    Stats& stats_;
    // Phase 10D: double-buffered fill presentation. current_buffers_ is the
    // committed state read by consume_result() / buffer() / current_busy();
    // next_buffers_ accumulates this cycle's try_write() fills and the staged
    // claim/release. commit() flips next_ -> current_.
    std::vector<LoadGatherBuffer> current_buffers_;
    std::vector<LoadGatherBuffer> next_buffers_;
    uint32_t rr_pointer_ = 0;
    // Single-port arbitration flag for the cache's line-to-gather-buffer
    // extraction port (§5.3 Port model: FILL > secondary > HIT). Writers
    // (cache.handle_responses, cache.drain_secondary_chain_head,
    // cache.process_load HIT path) all funnel through try_write(); the first
    // writer in a tick wins by reading next_port_claimed_, and the others
    // bail. commit() clears the flag at end-of-cycle so the next tick starts
    // unclaimed.
    bool next_port_claimed_ = false;

    // Phase M2: REGISTERED claim-request slot. claim() writes next_; commit()
    // flips into current_; evaluate() consumes current_ at top of next tick.
    GatherClaimRequest current_claim_request_;
    GatherClaimRequest next_claim_request_;

    // Phase 10D: REGISTERED buffer-release slot. consume_result() reads the
    // committed buffer (a pure read) and stages a release here; commit()
    // applies it. At most one buffer retires per cycle (single writeback
    // arbiter port), so a single-slot request is sufficient.
    GatherReleaseRequest next_release_;

    // Phase M4: REGISTERED has-result flag. commit() recomputes it from the
    // post-flip committed buffer state (true iff any buffer has busy &&
    // filled_count==WARP_SIZE); the writeback arbiter reads it via the
    // current_has_result() override.
    bool current_has_result_ = false;
};

} // namespace gpu_sim
