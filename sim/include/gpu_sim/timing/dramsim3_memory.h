#pragma once

#include "gpu_sim/config.h"
#include "gpu_sim/stats.h"
#include "gpu_sim/timing/memory_interface.h"

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
class DRAMSim3Memory : public ExternalMemoryInterface {
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

    const SimConfig& cfg_;
    Stats& stats_;

    std::unique_ptr<dramsim3::MemorySystem> mem_;

    uint32_t bytes_per_burst_;
    uint32_t line_size_;
    uint32_t chunks_per_line_;
    // The request FIFO is logically split into two reserved regions:
    //   - reads: num_mshrs * chunks_per_line slots, never starved by writes,
    //     guaranteeing submit_read always succeeds (since at most num_mshrs
    //     reads can be in flight by MSHR construction).
    //   - writes: write_buffer_depth * chunks_per_line slots — submit_write
    //     returns false when the write region is full, which the cache
    //     consumes as backpressure (drain_write_buffer pops only on success).
    // request_fifo_depth_ is exactly the sum of the two regions. Anything
    // larger is wasteful; anything smaller violates the architectural bound.
    uint32_t request_fifo_depth_;
    uint32_t write_region_capacity_;
    // Architectural bounds on the two completion queues, each asserted at
    // its push site:
    //   - read responses: at most one per in-flight MSHR, so num_mshrs.
    //   - write acks: the cache caps enqueued-but-unacked write-throughs at
    //     max_outstanding_writes (eager-wobbling-pizza.md Steps 1-2); the
    //     synthetic write ack is 1:1 with submit_write, so the write-ack
    //     queue holds a subset of those. The chunks_per_line cushion absorbs
    //     the bounded same-cycle multi-enqueue overshoot.
    uint32_t read_response_queue_capacity_;
    uint32_t write_ack_queue_capacity_;
    // Synthetic write-ack durability latency: DRAM cycles from a write being
    // issued to DRAM until its ack is released to the cache (SimConfig
    // dramsim3_write_commit_latency_tck). Models the DDR write-commit window
    // (~CWL + BL/2 + tWR, plus column spacing for a multi-chunk line).
    uint32_t write_commit_latency_tck_;

    // DRAM ticks per fabric tick. Fractional ratios accumulate in `phase_`.
    double ticks_per_fabric_;
    double phase_ = 0.0;

    // Bounded request FIFO (fabric clock producer; DRAM clock consumer).
    std::deque<PendingChunk> request_fifo_;
    // Number of write chunks currently in request_fifo_. Tracked explicitly
    // so submit_write can enforce the write-region bound in O(1).
    uint32_t write_chunks_in_fifo_ = 0;

    // Response FIFO (DRAM clock producer; fabric clock consumer). Reads only.
    std::deque<MemoryResponse> responses_;
    // Write-ack FIFO — synthetic write acks released after the commit
    // latency, drained unconditionally by the cache.
    std::deque<MemoryResponse> write_acks_;
    // Synthetic write acks awaiting their commit latency. One per submit_write,
    // sorted by release tick (submit order, constant latency); the front-most
    // ready entry is promoted into write_acks_ each evaluate().
    std::deque<PendingWriteAck> pending_write_acks_;

    // Per-MSHR read reassembly. Indexed by mshr_id; sized to cfg.num_mshrs.
    std::vector<ReadAssembly> read_assembly_;

    // Reverse map from a read chunk's byte address to its owning MSHR.
    std::unordered_map<uint64_t, uint32_t> read_chunk_to_mshr_;

    // Fabric-clock cycle counter, incremented once per evaluate(). Used to
    // tag in-flight reads with their submit cycle so per-request latency
    // (submit→response) can be accumulated into Stats on completion.
    uint64_t fabric_cycle_ = 0;

    // Total DRAMSim3 ClockTicks issued since construction or last reset.
    size_t dram_ticks_ = 0;
    // Peak queue depths observed; used by tests to assert the architectural
    // bounds are respected under stress.
    size_t max_response_queue_ = 0;
    size_t max_write_ack_queue_ = 0;

    // Phase M5: REGISTERED request slots.
    PendingMemoryRequest current_read_request_;
    PendingMemoryRequest next_read_request_;
    PendingMemoryRequest current_write_request_;
    PendingMemoryRequest next_write_request_;
};

} // namespace gpu_sim
