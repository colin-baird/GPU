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

// One gather buffer per resident warp, registered with the writeback arbiter
// as an ExecutionUnit source so its full-32-lane writebacks arbitrate alongside
// ALU/MUL/DIV/TLOOKUP results.
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
    bool try_write(uint32_t warp_id, uint32_t lane_mask,
                   const std::array<uint32_t, WARP_SIZE>& values,
                   GatherWriteSource source);

    // ExecutionUnit interface
    // Phase M2: evaluate() now applies a deferred claim from
    // current_claim_request_ if valid. Tick scheduling places this evaluate
    // before cache.evaluate() so that any same-cycle FILL or HIT write
    // observes the freshly-applied claim metadata (busy, dest_reg, etc.).
    // Phase 10B.0.5: empty body. LoadGatherBufferFile is the variable-latency
    // load-writeback path, not one of the issue/execute stages the 10B.0.5
    // double-buffering normalization targets — it is out of scope for that
    // phase. Its cross-cycle state (the REGISTERED claim-request and
    // has-result slots) is written by event-shaped methods and flipped at
    // commit(), not carried via a seeded next_*; the buffers_ vector is
    // mutated directly. TimingModel::tick() does not call this seed_next();
    // it exists only to satisfy the ExecutionUnit interface, and the empty
    // body leaves behavior byte-identical.
    void seed_next() override {}
    void evaluate() override;
    void commit() override;
    void reset() override;

    // Panic flush hook. Called at the commit-phase boundary when the panic
    // signal becomes active. Delegates to reset().
    void flush();
    // Phase M4: REGISTERED has-result. The base interface keeps the
    // `next_has_result()` name for compatibility with other ExecutionUnit
    // overrides that are still mid-evaluate live reads (ALU/MUL/DIV/TLOOKUP).
    // For the gather buffer, the implementation returns
    // `current_has_result_` (latched at commit when filled_count reaches
    // WARP_SIZE for any buffer); the writeback arbiter and timing-model
    // drain checks therefore see committed state for this source.
    bool next_has_result() const override;
    // Canonical accessor name for the REGISTERED contract. Returns the same
    // value as next_has_result(); preferred at new call sites.
    bool current_has_result() const { return current_has_result_; }
    WritebackEntry consume_result() override;
    ExecUnit get_type() const override { return ExecUnit::LDST; }
    // LoadGatherBufferFile is a writeback source (consumed by the writeback
    // arbiter), never a scheduler dispatch target. The scheduler routes
    // LDST instructions to LdStUnit, so current_busy() is a constant false
    // that the scheduler never queries.
    bool current_busy() const override { return false; }

    uint32_t num_buffers() const { return num_warps_; }
    const LoadGatherBuffer& buffer(uint32_t warp_id) const { return buffers_[warp_id]; }

private:
    uint32_t num_warps_;
    Stats& stats_;
    std::vector<LoadGatherBuffer> buffers_;
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

    // Phase M4: REGISTERED has-result flag. try_write recomputes the staged
    // value (true iff any buffer has busy && filled_count==WARP_SIZE);
    // commit() flips next_has_result_ → current_has_result_; the writeback
    // arbiter reads current_has_result_ via next_has_result() (base interface
    // name preserved). consume_result() releases the buffer directly; the
    // next try_write or commit recomputes the flag.
    bool current_has_result_ = false;
    bool next_has_result_ = false;
};

} // namespace gpu_sim
