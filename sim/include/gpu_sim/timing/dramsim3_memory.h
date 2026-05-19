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
// Phase 6 (reg.h migration): mirrors FixedLatencyMemory's REGISTERED request
// slot wrapping. The DRAM-scheduling internals (request_fifo_, in-flight
// reassembly, response/write-ack queues, fabric/DRAM clock counters) are
// internal scheduling/instrumentation state, NOT REGISTERED double-buffered
// pairs, so they stay plain — see the annotations on each member.
class DRAMSim3Memory : public ExternalMemoryInterface, public RegisteredStage {
public:
    DRAMSim3Memory(const SimConfig& cfg, Stats& stats);
    ~DRAMSim3Memory() override;

    void evaluate() override;
    void commit() override;
    void reset() override;

    // Phase M5: REGISTERED forward request path.
    void set_next_read_request(uint32_t line_addr, uint32_t mshr_id) override;
    void set_next_write_request(uint32_t line_addr) override;
    // COMBINATIONAL backward stall — true when the write region of the
    // request FIFO can't accept another write's worth of chunks. Reads
    // are never stalled (the read region is sized to num_mshrs * chunks_per_line).
    bool next_request_stall() const override;

    bool submit_read(uint32_t line_addr, uint32_t mshr_id) override;
    bool submit_write(uint32_t line_addr) override;

    bool current_has_response() const override { return !responses_.empty(); }
    MemoryResponse get_response() override;
    bool current_has_write_ack() const override { return !write_acks_.empty(); }
    MemoryResponse get_write_ack() override;
    size_t write_ack_count() const override { return write_acks_.size(); }
    bool is_idle() const override;
    size_t in_flight_count() const override;
    size_t response_count() const override { return responses_.size(); }

    // Snapshot helpers for tests.
    size_t request_fifo_size() const { return request_fifo_.size(); }
    uint32_t chunks_per_line() const { return chunks_per_line_; }
    size_t dram_ticks() const { return dram_ticks_; }
    size_t max_observed_response_queue() const { return max_response_queue_; }
    uint32_t read_response_queue_capacity() const { return read_response_queue_capacity_; }
    size_t max_observed_write_ack_queue() const { return max_write_ack_queue_; }
    uint32_t write_ack_queue_capacity() const { return write_ack_queue_capacity_; }

private:
    struct PendingChunk {
        uint64_t chunk_byte_addr;
        uint32_t line_addr;
        uint32_t mshr_id;   // unused for writes
        bool     is_write;
    };

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

    // ── Internal scheduling state (plain, not REGISTERED) ──────────────────
    //
    // Phase 6 (reg.h migration) — request_fifo_ exception, analogous to the
    // Phase 3 ldst addr_gen_fifo_ exception (commit bed4043). RegFifo<T> is
    // NOT the right abstraction here: the producer and consumer co-occur
    // within a single evaluate(). At the top of evaluate(), the REGISTERED
    // current_*_request_ slot is drained by calling submit_read/submit_write,
    // which push chunks straight into request_fifo_. Later in the SAME
    // evaluate(), the `while (phase_ >= 1.0)` loop pops from
    // request_fifo_.front() to issue chunks into DRAMSim3. RegFifo's
    // commit-time pop-then-push would defer the staged push until after
    // evaluate() returns, denying the consumer this cycle's just-submitted
    // chunks — a real cycle-of-latency behavior change. Test paths also
    // call submit_read/submit_write directly and immediately call evaluate(),
    // expecting same-cycle DRAM admission availability; RegFifo would
    // require an intervening commit(). The deque + explicit
    // write_chunks_in_fifo_ occupancy counter encodes the in-evaluate
    // producer→consumer pattern with no behavior change.
    std::deque<PendingChunk> request_fifo_;   // internal scheduling queue (Phase-3-style RegFifo exception)
    // Number of write chunks currently in request_fifo_. Tracked explicitly
    // so submit_write can enforce the write-region bound in O(1). Mirrors
    // request_fifo_'s plain-deque kind.
    uint32_t write_chunks_in_fifo_ = 0;       // internal scheduling counter

    // Response FIFO (DRAM clock producer; fabric clock consumer). Reads only.
    std::deque<MemoryResponse> responses_;    // internal scheduling queue
    // Write-ack FIFO — synthetic write acks released after the commit
    // latency, drained unconditionally by the cache.
    std::deque<MemoryResponse> write_acks_;   // internal scheduling queue
    // Synthetic write acks awaiting their commit latency. One per submit_write,
    // sorted by release tick (submit order, constant latency); the front-most
    // ready entry is promoted into write_acks_ each evaluate().
    std::deque<PendingWriteAck> pending_write_acks_;  // internal scheduling queue

    // Per-MSHR read reassembly. Indexed by mshr_id; sized to cfg.num_mshrs.
    std::vector<ReadAssembly> read_assembly_; // internal scheduling state

    // Reverse map from a read chunk's byte address to its owning MSHR.
    std::unordered_map<uint64_t, uint32_t> read_chunk_to_mshr_;  // internal scheduling state

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
    // Phase 6 (reg.h migration): wrapped as Reg<PendingMemoryRequest> and
    // enrolled via RegisteredStage::register_state. Memoryless-consumer
    // contract — evaluate() drains current() into the request FIFO and
    // invalidates the committed slot via the documented current_mut() escape
    // hatch; commit() drives commit_all() and then explicitly clears the
    // staged slot via set_next(PendingMemoryRequest{}). The backend opts
    // out of the seed phase for the same reason FixedLatencyMemory does
    // (and the same reason L1Cache::load_cmd_/store_cmd_ do in Phase 5a):
    // auto-seeding next from current would re-latch the consumed request.
    Reg<PendingMemoryRequest> read_request_;
    Reg<PendingMemoryRequest> write_request_;
};

} // namespace gpu_sim
