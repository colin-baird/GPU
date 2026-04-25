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

    bool submit_read(uint32_t line_addr, uint32_t mshr_id) override;
    bool submit_write(uint32_t line_addr) override;

    bool has_response() const override { return !responses_.empty(); }
    MemoryResponse get_response() override;
    bool is_idle() const override;
    size_t in_flight_count() const override;
    size_t response_count() const override { return responses_.size(); }

    // Snapshot helpers for tests.
    size_t request_fifo_size() const { return request_fifo_.size(); }
    uint32_t chunks_per_line() const { return chunks_per_line_; }
    size_t dram_ticks() const { return dram_ticks_; }
    size_t max_observed_response_queue() const { return max_response_queue_; }
    uint32_t response_queue_capacity() const { return response_queue_capacity_; }

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
    };

    struct WriteAssembly {
        uint16_t chunks_remaining = 0;
    };

    void rebuild_memory_system();
    void on_read_complete(uint64_t hex_addr);
    void on_write_complete(uint64_t hex_addr);

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
    // Architectural bound on the response queue depth: at most one read
    // response per in-flight MSHR plus the writes that may finalize between
    // two consecutive cache drains (bounded by write_buffer_depth distinct
    // line slots plus a chunks_per_line cushion for in-evaluate completions).
    // Push sites assert against this to catch any violation immediately.
    uint32_t response_queue_capacity_;

    // DRAM ticks per fabric tick. Fractional ratios accumulate in `phase_`.
    double ticks_per_fabric_;
    double phase_ = 0.0;

    // Bounded request FIFO (fabric clock producer; DRAM clock consumer).
    std::deque<PendingChunk> request_fifo_;
    // Number of write chunks currently in request_fifo_. Tracked explicitly
    // so submit_write can enforce the write-region bound in O(1).
    uint32_t write_chunks_in_fifo_ = 0;

    // Response FIFO (DRAM clock producer; fabric clock consumer).
    std::deque<MemoryResponse> responses_;

    // Per-MSHR read reassembly. Indexed by mshr_id; sized to cfg.num_mshrs.
    std::vector<ReadAssembly> read_assembly_;

    // Per-line write reassembly. Multiple in-flight writes to the same line
    // share one slot.
    std::unordered_map<uint32_t, WriteAssembly> write_assembly_;

    // Reverse maps from chunk byte address (issued to DRAMSim3) to its owner.
    std::unordered_map<uint64_t, uint32_t> read_chunk_to_mshr_;
    std::unordered_map<uint64_t, uint32_t> write_chunk_to_line_;

    // Total DRAMSim3 ClockTicks issued since construction or last reset.
    size_t dram_ticks_ = 0;
    // Peak response queue depth observed; used by tests to assert the
    // architectural bound is respected under stress.
    size_t max_response_queue_ = 0;
};

} // namespace gpu_sim
