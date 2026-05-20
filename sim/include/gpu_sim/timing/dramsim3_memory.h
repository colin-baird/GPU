#pragma once

#include "gpu_sim/config.h"
#include "gpu_sim/stats.h"
#include "gpu_sim/timing/memory_interface.h"
#include "gpu_sim/timing/reg.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

namespace dramsim3 { class MemorySystem; }

namespace gpu_sim {

// External-memory backend backed by DRAMSim3. Sits behind the same
// `ExternalMemoryInterface` surface used by the cache and is selected by
// `SimConfig::memory_backend == "dramsim3"`. Models the DE-10 Nano DDR3 with
// an asynchronous fabric/DRAM clock crossing and chunked cache-line transfers.
//
// Phase 5 (sparkling-dazzling-starfish.md): the prior implementation
// pushed onto request_fifo_ at the top of evaluate() and the
// `while (phase_ >= 1.0)` drain loop popped from the same fifo later in
// the same evaluate() — fabric-clock push and DRAM-clock pop collapsed
// into one body, hiding the one-fabric-cycle CDC traversal latency the
// FIFO presents in real RTL. The class is now structured as two clock
// stages — a fabric-clock half (evaluate_fabric) and a DRAM-clock half
// (evaluate_dram) — sequenced at independent positions in
// TimingModel::tick()'s sweep. request_fifo_ moved to TimingModel as a
// RegFifo<DRAMSim3PendingChunk> committed in the dedicated ungated
// cross-stage pass; the fabric-clock half stages chunk pushes, the
// DRAM-clock half stages chunk pops. The internal scheduling state
// (read_assembly_, read_chunk_to_mshr_, pending_write_acks_,
// write_chunks_in_fifo_) is now wrapped in Reg<T> and enrolled in the
// class's RegisteredStage so commit_all() flips each at the cycle
// boundary; fabric/DRAM partitioning is enforced by call discipline (each
// member is mutated only by the half that owns it). The shared
// PulseReg<PendingMemoryRequest> request slots (read_request_,
// write_request_) live on the fabric half. Completion responses /
// write-acks stage onto the TimingModel-owned cross-stage RegFifos
// (mem_responses_ / mem_write_acks_) — the Phase-4 slots — from the
// DRAM-clock half; the cache reads current_response_front() last cycle's
// committed completion, the documented CDC latency.
struct DRAMSim3PendingChunk {
    uint64_t chunk_byte_addr = 0;
    uint32_t line_addr = 0;
    uint32_t mshr_id = 0;   // unused for writes
    bool     is_write = false;
};

class DRAMSim3Memory : public ExternalMemoryInterface, public RegisteredStage {
public:
    DRAMSim3Memory(const SimConfig& cfg, Stats& stats);
    ~DRAMSim3Memory() override;

    // Phase 5 (sparkling-dazzling-starfish.md): stage split. The fabric
    // half drains the PulseReg<PendingMemoryRequest> request slots and
    // stages chunk pushes onto the TimingModel-owned request_fifo_; the
    // DRAM half walks the phase_/ClockTick loop, stages per-tick pops on
    // request_fifo_, and stages mem_responses_ / mem_write_acks_ pushes
    // from DRAMSim3 completion callbacks. Both halves are ungated (no
    // writeback stall on memory backends), so a single commit() and a
    // single reset() suffice — the registered set holds the union of
    // both halves' Reg state.
    void evaluate_fabric() override;
    void evaluate_dram() override;
    void commit() override;
    void reset() override;

    // Phase 5 (sparkling-dazzling-starfish.md): cross-stage response FIFO
    // back-pointers — DRAMSim3 now uses the TimingModel-owned slots
    // (DRAM-clock-side stage_push from on_read_complete / synthetic ack
    // release; fabric-clock-side stage_pop from cache.handle_responses).
    // The cache reads current_response_front() — last cycle's committed
    // completion — surfacing the documented one-cycle CDC traversal
    // latency.
    void set_response_queues(RegFifo<MemoryResponse>* responses,
                             RegFifo<MemoryResponse>* write_acks) override {
        mem_responses_ = responses;
        mem_write_acks_ = write_acks;
    }

    // Phase 5 (sparkling-dazzling-starfish.md): cross-stage request FIFO
    // back-pointer. The FIFO is owned by TimingModel and committed in the
    // dedicated ungated cross-stage pass alongside addr_gen_fifo_ /
    // mem_responses_ / mem_write_acks_. nullptr-tolerant for unit-test
    // fixtures that own a local RegFifo and commit it directly.
    void set_request_fifo(RegFifo<DRAMSim3PendingChunk>* fifo) {
        request_fifo_ = fifo;
    }

    // Phase M5: REGISTERED forward request path.
    void set_next_read_request(uint32_t line_addr, uint32_t mshr_id) override;
    void set_next_write_request(uint32_t line_addr) override;
    // COMBINATIONAL backward stall — true when the write region of the
    // request FIFO can't accept another write's worth of chunks. Reads
    // are never stalled (the read region is sized to num_mshrs * chunks_per_line).
    bool next_request_stall() const override;

    bool submit_read(uint32_t line_addr, uint32_t mshr_id) override;
    bool submit_write(uint32_t line_addr) override;

    // Phase 5 (sparkling-dazzling-starfish.md): peek + staged-pop on the
    // TimingModel-owned cross-stage RegFifos. The cache reads
    // current_response_front() — last cycle's committed completion (the
    // natural one-cycle FIFO latency a real hardware FIFO presents) — and
    // calls stage_response_pop() to claim it; the pop applies at
    // TimingModel::commit_cross_stage_fifos(). nullptr-tolerant for
    // fixtures that exercise the backend with a local RegFifo.
    bool current_has_response() const override {
        return mem_responses_ && !mem_responses_->current_empty();
    }
    const MemoryResponse& current_response_front() const override {
        return mem_responses_->current_front();
    }
    void stage_response_pop() override {
        if (mem_responses_) mem_responses_->stage_pop();
    }
    bool current_has_write_ack() const override {
        return mem_write_acks_ && !mem_write_acks_->current_empty();
    }
    const MemoryResponse& current_write_ack_front() const override {
        return mem_write_acks_->current_front();
    }
    void stage_write_ack_pop() override {
        if (mem_write_acks_) mem_write_acks_->stage_pop();
    }
    size_t write_ack_count() const override {
        return mem_write_acks_ ? mem_write_acks_->current_size() : 0u;
    }
    bool is_idle() const override;
    size_t in_flight_count() const override;
    size_t response_count() const override {
        return mem_responses_ ? mem_responses_->current_size() : 0u;
    }

    // Snapshot helpers for tests.
    size_t request_fifo_size() const {
        return request_fifo_ ? request_fifo_->current_size() : 0u;
    }
    uint32_t chunks_per_line() const { return chunks_per_line_; }
    size_t dram_ticks() const { return dram_ticks_; }
    size_t max_observed_response_queue() const { return max_response_queue_; }
    uint32_t read_response_queue_capacity() const { return read_response_queue_capacity_; }
    size_t max_observed_write_ack_queue() const { return max_write_ack_queue_; }
    uint32_t write_ack_queue_capacity() const { return write_ack_queue_capacity_; }

private:
    // PendingChunk has been promoted to the public header scope above as
    // DRAMSim3PendingChunk so TimingModel can hold a
    // RegFifo<DRAMSim3PendingChunk> for the cross-stage request_fifo_.

    struct ReadAssembly {
        uint32_t line_addr        = 0;
        uint16_t chunks_remaining = 0;
        bool     active           = false;
        uint64_t submit_cycle     = 0;
    };

    // One synthetic write ack, scheduled per submit_write. DRAMSim3's own
    // write callback fires at submit+1 (it models a posted write) and folds
    // multiple in-flight same-line writes into one event — neither faithful
    // to write durability nor 1:1 with write-throughs. So the wrapper
    // synthesizes a write ack per submit_write, released a parameterized
    // DRAM-cycle commit latency after the write is issued to DRAM. This is
    // 1:1 with the cache's outstanding-write counter by construction.
    struct PendingWriteAck {
        uint32_t line_addr;
        uint64_t release_dram_tick;
    };

    void rebuild_memory_system();
    void on_read_complete(uint64_t hex_addr);

    const SimConfig& cfg_;  // config (back-pointer)
    Stats& stats_;          // config (back-pointer)

    std::unique_ptr<dramsim3::MemorySystem> mem_;  // config (constructed once / on reset)

    // ── Config (const after construction) ──────────────────────────────────
    uint32_t bytes_per_burst_;   // config
    uint32_t line_size_;         // config
    uint32_t chunks_per_line_;   // config
    // The request FIFO is logically split into two reserved regions:
    //   - reads: num_mshrs * chunks_per_line slots, never starved by writes,
    //     guaranteeing submit_read always succeeds (since at most num_mshrs
    //     reads can be in flight by MSHR construction).
    //   - writes: write_buffer_depth * chunks_per_line slots — submit_write
    //     returns false when the write region is full, which the cache
    //     consumes as backpressure (drain_write_buffer pops only on success).
    // request_fifo_depth_ is exactly the sum of the two regions. Anything
    // larger is wasteful; anything smaller violates the architectural bound.
    uint32_t request_fifo_depth_;    // config
    uint32_t write_region_capacity_; // config
    // Architectural bounds on the two completion queues, each asserted at
    // its push site:
    //   - read responses: at most one per in-flight MSHR, so num_mshrs.
    //   - write acks: the cache caps enqueued-but-unacked write-throughs at
    //     max_outstanding_writes (eager-wobbling-pizza.md Steps 1-2); the
    //     synthetic write ack is 1:1 with submit_write, so the write-ack
    //     queue holds a subset of those. The chunks_per_line cushion absorbs
    //     the bounded same-cycle multi-enqueue overshoot.
    uint32_t read_response_queue_capacity_;  // config
    uint32_t write_ack_queue_capacity_;      // config
    // Synthetic write-ack durability latency: DRAM cycles from a write being
    // issued to DRAM until its ack is released to the cache (SimConfig
    // dramsim3_write_commit_latency_tck). Models the DDR write-commit window
    // (~CWL + BL/2 + tWR, plus column spacing for a multi-chunk line).
    uint32_t write_commit_latency_tck_;      // config

    // DRAM ticks per fabric tick. Fractional ratios accumulate in `phase_`.
    double ticks_per_fabric_;          // config
    double phase_ = 0.0;               // sim-instrumentation (fractional cross-clock accumulator)

    // ── Internal scheduling state ──────────────────────────────────────────
    //
    // Phase 5 (sparkling-dazzling-starfish.md): the cross-stage request
    // FIFO (was `std::deque<PendingChunk> request_fifo_`) moves to
    // TimingModel as `RegFifo<DRAMSim3PendingChunk>` and is committed in
    // the dedicated ungated cross-stage pass; the back-pointer below
    // references it. Both clock halves use the multi-push / multi-pop
    // extensions on RegFifo (the fabric half stages up to chunks_per_line
    // pushes per submit; the DRAM half may stage multiple pops per fabric
    // cycle when ticks_per_fabric_ > 1).
    RegFifo<DRAMSim3PendingChunk>* request_fifo_ = nullptr;  // timing-naming-allow: back-pointer to TimingModel-owned cross-stage RegFifo (Phase-3/4 pattern); the FIFO itself is committed at TimingModel::commit_cross_stage_fifos().
    RegFifo<MemoryResponse>* mem_responses_ = nullptr;       // timing-naming-allow: back-pointer to TimingModel-owned cross-stage RegFifo (Phase-3/4 pattern).
    RegFifo<MemoryResponse>* mem_write_acks_ = nullptr;      // timing-naming-allow: back-pointer to TimingModel-owned cross-stage RegFifo (Phase-3/4 pattern).

    // Number of write chunks currently in request_fifo_. Tracked
    // explicitly so submit_write can enforce the write-region bound in
    // O(1). Phase 5 (sparkling-dazzling-starfish.md): wrapped as
    // Reg<uint32_t>, mutated in fabric-clock half on push, in DRAM-clock
    // half on pop, and committed atomically with the rest of the
    // backend's registered state. The fabric and DRAM halves both write
    // next_mut() through different code paths in the same tick; that is
    // legal because both stages are ungated and the cross-half visibility
    // is intra-tick (the value is committed at end-of-tick, so an
    // assertion in submit_write that reads current() reflects last-tick's
    // committed value — the natural one-fabric-cycle freshness).
    Reg<uint32_t> write_chunks_in_fifo_;  // fabric half: incremented on submit_write; DRAM half: decremented on chunk issue

    // Synthetic write acks awaiting their commit latency. One per
    // submit_write, sorted by release tick (submit order, constant
    // latency); the front-most ready entry is promoted into the write-ack
    // RegFifo each DRAM-clock evaluate.
    // Phase 5: Reg<std::deque<PendingWriteAck>>, owned by the DRAM-clock
    // half (with the per-submit push staged from the fabric half — same
    // intra-tick next_mut() shape as write_chunks_in_fifo_).
    Reg<std::deque<PendingWriteAck>> pending_write_acks_;

    // Per-MSHR read reassembly. Indexed by mshr_id; sized to cfg.num_mshrs.
    // Phase 5: Reg<std::vector<ReadAssembly>>, owned by the DRAM-clock
    // half (the submit_read entry that activates a slot is also written
    // by the fabric half — intra-tick next_mut() shape).
    Reg<std::vector<ReadAssembly>> read_assembly_;

    // Reverse map from a read chunk's byte address to its owning MSHR.
    // Phase 5: Reg<std::unordered_map<uint64_t, uint32_t>>, DRAM-clock half.
    Reg<std::unordered_map<uint64_t, uint32_t>> read_chunk_to_mshr_;

    // ── Simulator instrumentation (not modeled clocked hardware) ───────────
    // Fabric-clock cycle counter, incremented once per evaluate(). Used to
    // tag in-flight reads with their submit cycle so per-request latency
    // (submit→response) can be accumulated into Stats on completion.
    uint64_t fabric_cycle_ = 0;     // sim-instrumentation

    // Total DRAMSim3 ClockTicks issued since construction or last reset.
    size_t dram_ticks_ = 0;         // sim-instrumentation
    // Peak queue depths observed; used by tests to assert the architectural
    // bounds are respected under stress.
    size_t max_response_queue_ = 0; // sim-instrumentation
    size_t max_write_ack_queue_ = 0;// sim-instrumentation

    // ── REGISTERED state ───────────────────────────────────────────────────
    // Phase 4 of current_mut() elimination (Pattern 3): PulseReg<T> request
    // slots. PulseReg::commit() defaults next_ to T{} after the flip; cache
    // overrides by calling set_next_*_request during its evaluate. No
    // mid-cycle current_mut() write, no tail-of-commit set_next(T{}) — the
    // commit-time reset inside the type encodes the memoryless contract.
    PulseReg<PendingMemoryRequest> read_request_;
    PulseReg<PendingMemoryRequest> write_request_;
};

} // namespace gpu_sim
