#include "gpu_sim/timing/dramsim3_memory.h"

#include "dramsim3.h"

#include <cassert>
#include <filesystem>
#include <stdexcept>
#include <utility>

namespace gpu_sim {

DRAMSim3Memory::DRAMSim3Memory(const SimConfig& cfg, Stats& stats)
    : cfg_(cfg),
      stats_(stats),
      bytes_per_burst_(cfg.dramsim3_bytes_per_burst),
      line_size_(cfg.cache_line_size_bytes),
      chunks_per_line_(cfg.cache_line_size_bytes / cfg.dramsim3_bytes_per_burst),
      request_fifo_depth_(cfg.dramsim3_request_fifo_depth),
      write_region_capacity_(cfg.write_buffer_depth *
                             (cfg.cache_line_size_bytes /
                              cfg.dramsim3_bytes_per_burst)),
      read_response_queue_capacity_(cfg.num_mshrs),
      write_ack_queue_capacity_(cfg.max_outstanding_writes +
                                (cfg.cache_line_size_bytes /
                                 cfg.dramsim3_bytes_per_burst)),
      write_commit_latency_tck_(cfg.dramsim3_write_commit_latency_tck),
      ticks_per_fabric_(cfg.dram_clock_mhz / cfg.fpga_clock_mhz) {
    if (cfg.dramsim3_config_path.empty()) {
        throw std::invalid_argument(
            "DRAMSim3Memory requires a non-empty dramsim3_config_path");
    }
    if (chunks_per_line_ == 0) {
        throw std::invalid_argument(
            "DRAMSim3Memory: cache_line_size_bytes < dramsim3_bytes_per_burst");
    }
    // Phase 5 (sparkling-dazzling-starfish.md): size the per-MSHR
    // read-assembly vector once, then enroll every Reg in the registered
    // set so commit_all() / reset_all() drive them uniformly. Both clock
    // halves (evaluate_fabric / evaluate_dram) share the same registered
    // set since neither is writeback-stall-gated; the fabric/DRAM
    // partitioning is enforced by call discipline, not by separate commits.
    read_assembly_.initialize(std::vector<ReadAssembly>(cfg.num_mshrs));
    register_state(&read_request_, &write_request_,
                   &write_chunks_in_fifo_, &pending_write_acks_,
                   &read_assembly_, &read_chunk_to_mshr_);
    rebuild_memory_system();
}

DRAMSim3Memory::~DRAMSim3Memory() = default;

void DRAMSim3Memory::rebuild_memory_system() {
    mem_.reset();
    // DRAMSim3 prints a "WARNING: Output directory ... not exists!" to stdout
    // when the dir is missing — that contaminates --json output. Pre-create it
    // so the warning never fires. Errors are intentionally swallowed: if the
    // dir can't be created, DRAMSim3 falls back to cwd and emits its warning.
    std::error_code ec;
    std::filesystem::create_directories(cfg_.dramsim3_output_dir, ec);
    auto rcb = [this](uint64_t a) { on_read_complete(a); };
    // DRAMSim3's write callback fires at submit+1 (it models a posted write)
    // and is address-only — it cannot drive a 1:1, durability-faithful write
    // ack. Writes are still issued to DRAMSim3 for bank/bus contention, but
    // the cache-facing write ack is synthesized in submit_write/evaluate;
    // the write callback itself is intentionally a no-op.
    auto wcb = [](uint64_t) {};
    mem_.reset(dramsim3::GetMemorySystem(cfg_.dramsim3_config_path,
                                         cfg_.dramsim3_output_dir,
                                         std::move(rcb),
                                         std::move(wcb)));
}

bool DRAMSim3Memory::submit_read(uint32_t line_addr, uint32_t mshr_id) {
    // Phase 5 (sparkling-dazzling-starfish.md): runs on the fabric clock.
    // Stages chunk pushes on the TimingModel-owned cross-stage
    // request_fifo_ (multi-push via the Phase 5 stage_push extension);
    // the DRAM-clock half sees the new chunks one fabric cycle later via
    // request_fifo_->current()[pops_staged()], the documented CDC
    // traversal latency.
    auto& assembly = read_assembly_.next_mut();
    if (mshr_id >= assembly.size()) {
        throw std::out_of_range("DRAMSim3Memory: mshr_id out of range");
    }
    // Architectural invariant: at most num_mshrs reads can be in flight at
    // once, and the request FIFO reserves num_mshrs * chunks_per_line slots
    // for reads. The bound is checked against the committed size plus any
    // pushes already staged this fabric cycle (multi-line submits within
    // one cycle from a stressed path).
    const std::size_t fifo_occupancy =
        (request_fifo_ ? request_fifo_->current_size() + request_fifo_->pushes_staged() : 0u);
    assert(fifo_occupancy + chunks_per_line_ <= request_fifo_depth_ &&
           "DRAMSim3Memory request FIFO would overflow on submit_read; "
           "writes must be confined to the write region (submit_write enforces "
           "this) so reads always have their reserved slots");
    if (fifo_occupancy + chunks_per_line_ > request_fifo_depth_) {
        return false;
    }
    auto& a = assembly[mshr_id];
    if (a.active) {
        // The cache should not reuse an MSHR id while one is in flight; bail
        // out rather than corrupt state.
        return false;
    }
    a.line_addr        = line_addr;
    a.chunks_remaining = static_cast<uint16_t>(chunks_per_line_);
    a.active           = true;
    a.submit_cycle     = fabric_cycle_;

    const uint64_t base = static_cast<uint64_t>(line_addr) * line_size_;
    for (uint32_t i = 0; i < chunks_per_line_; ++i) {
        DRAMSim3PendingChunk chunk;
        chunk.chunk_byte_addr = base + static_cast<uint64_t>(i) * bytes_per_burst_;
        chunk.line_addr = line_addr;
        chunk.mshr_id = mshr_id;
        chunk.is_write = false;
        if (request_fifo_) request_fifo_->stage_push(chunk);
    }
    stats_.external_memory_reads++;
    return true;
}

bool DRAMSim3Memory::submit_write(uint32_t line_addr) {
    // Phase 5 (sparkling-dazzling-starfish.md): fabric-clock half. The
    // write-region bound is checked against the committed write-chunk
    // count plus any same-cycle pushes that have not yet committed — those
    // are tracked through the staged write_chunks_in_fifo_ counter
    // (next_mut(), still pre-commit). Writes are confined to
    // write_region_capacity_ * chunks_per_line slots; beyond that, return
    // false so the cache holds the entry in its write_buffer and retries.
    const uint32_t staged_write_chunks = write_chunks_in_fifo_.next();
    if (staged_write_chunks + chunks_per_line_ > write_region_capacity_) {
        return false;
    }
    const uint64_t base = static_cast<uint64_t>(line_addr) * line_size_;
    for (uint32_t i = 0; i < chunks_per_line_; ++i) {
        DRAMSim3PendingChunk chunk;
        chunk.chunk_byte_addr = base + static_cast<uint64_t>(i) * bytes_per_burst_;
        chunk.line_addr = line_addr;
        chunk.mshr_id = 0;
        chunk.is_write = true;
        if (request_fifo_) request_fifo_->stage_push(chunk);
    }
    write_chunks_in_fifo_.next_mut() += chunks_per_line_;
    stats_.external_memory_writes++;
    // Schedule one synthetic write ack for this write-through, released
    // write_commit_latency_tck_ DRAM cycles from now (the write being issued
    // to DRAM). Exactly one ack per submit_write — 1:1 with the cache's
    // outstanding-write counter, with no same-line folding.
    pending_write_acks_.next_mut().push_back(
        {line_addr, dram_ticks_ + write_commit_latency_tck_});
    return true;
}

void DRAMSim3Memory::set_next_read_request(uint32_t line_addr, uint32_t mshr_id) {
    auto& rr = read_request_.next_mut();
    rr.valid = true;
    rr.line_addr = line_addr;
    rr.mshr_id = mshr_id;
}

void DRAMSim3Memory::set_next_write_request(uint32_t line_addr) {
    auto& wr = write_request_.next_mut();
    wr.valid = true;
    wr.line_addr = line_addr;
}

bool DRAMSim3Memory::next_request_stall() const {
    // Stall when the staged write-chunk count of the request FIFO can't
    // accept another line's worth of chunks. The cache reads this signal
    // backward in the same cycle from drain_write_buffer (which runs
    // AFTER mem.evaluate_fabric in the sweep), so the staged value
    // already reflects any chunks added by the fabric half this cycle —
    // RTL-faithful (this is a combinational backward signal, asserted
    // after the producer's evaluate). Reads have their own reserved
    // region and are never stalled.
    return write_chunks_in_fifo_.next() + chunks_per_line_ > write_region_capacity_;
}

void DRAMSim3Memory::evaluate_fabric() {
    // Phase 5 (sparkling-dazzling-starfish.md): fabric-clock half. Drain
    // current_*_request_ into staged request_fifo_ pushes. No mid-cycle Q
    // write — PulseReg<PendingMemoryRequest> defaults to T{} at the next
    // commit via its post-flip reset (see reg.h).
    if (read_request_.current().valid) {
        submit_read(read_request_.current().line_addr,
                    read_request_.current().mshr_id);
    }
    if (write_request_.current().valid) {
        submit_write(write_request_.current().line_addr);
    }
    ++fabric_cycle_;
}

void DRAMSim3Memory::evaluate_dram() {
    // Phase 5 (sparkling-dazzling-starfish.md): DRAM-clock half. Walks
    // the phase_/ClockTick loop using committed-state reads of
    // request_fifo_ (last fabric cycle's pushes). Stages multi-pops on
    // request_fifo_, multi-pushes on mem_responses_ / mem_write_acks_.
    // on_read_complete (invoked by mem_->ClockTick) writes into
    // read_assembly_.next_mut() and stages a push on mem_responses_.
    phase_ += ticks_per_fabric_;
    while (phase_ >= 1.0) {
        // Peek at the next chunk to issue: committed FIFO head minus the
        // pops staged so far this fabric cycle. The multi-pop counter on
        // RegFifo (Phase 5a) is what makes this composable across N DRAM
        // ticks per fabric cycle.
        if (request_fifo_) {
            const std::size_t already_staged = request_fifo_->pops_staged();
            const auto& committed = request_fifo_->current();
            if (already_staged < committed.size()) {
                const auto& c = committed[already_staged];
                if (mem_->WillAcceptTransaction(c.chunk_byte_addr, c.is_write)) {
                    if (c.is_write) {
                        // One chunk consumed: the staged write-chunk
                        // counter decrements by one. The cache reads
                        // committed state and only sees this change after
                        // the fabric-cycle commit.
                        write_chunks_in_fifo_.next_mut() -= 1u;
                    } else {
                        read_chunk_to_mshr_.next_mut()[c.chunk_byte_addr] = c.mshr_id;
                    }
                    mem_->AddTransaction(c.chunk_byte_addr, c.is_write);
                    request_fifo_->stage_pop();
                }
            }
        }
        mem_->ClockTick();
        ++dram_ticks_;
        phase_ -= 1.0;
    }

    // Release synthetic write acks whose commit latency has elapsed.
    // pending_write_acks_ is sorted by release tick; a front-drain is correct.
    // Phase 5: writes to mem_write_acks_ go through stage_push (multi-push
    // safe). pending_write_acks_ mutated through next_mut.
    auto& pending = pending_write_acks_.next_mut();
    std::size_t acks_pushed_this_cycle = 0;
    while (!pending.empty() && pending.front().release_dram_tick <= dram_ticks_) {
        assert(mem_write_acks_ &&
               "DRAMSim3Memory: mem_write_acks_ back-pointer required for ack release");
        MemoryResponse ack{pending.front().line_addr, 0, true};
        mem_write_acks_->stage_push(ack);
        ++acks_pushed_this_cycle;
        pending.pop_front();
    }
    // Track the queue-depth peak observed (committed size + staged
    // pushes, since the staged pushes are what would land at commit).
    if (mem_write_acks_) {
        const std::size_t depth =
            mem_write_acks_->current_size() + acks_pushed_this_cycle;
        assert(depth <= write_ack_queue_capacity_ &&
               "DRAMSim3Memory write-ack queue overflow on synthetic release");
        if (depth > max_write_ack_queue_) max_write_ack_queue_ = depth;
    }
}

void DRAMSim3Memory::commit() {
    // Phase 5 (sparkling-dazzling-starfish.md): commit_all() flips every
    // enrolled Reg in lockstep — both halves' registered state, the
    // PulseReg request slots, and the internal scheduling Reg<containers>.
    // The cross-stage RegFifos (request_fifo_, mem_responses_,
    // mem_write_acks_) are owned by TimingModel and committed in
    // TimingModel::commit_cross_stage_fifos(); never here.
    commit_all();
}

bool DRAMSim3Memory::is_idle() const {
    if (request_fifo_ && !request_fifo_->current_empty()) return false;
    if (mem_responses_ && !mem_responses_->current_empty()) return false;
    if (mem_write_acks_ && !mem_write_acks_->current_empty()) return false;
    if (!pending_write_acks_.current().empty()) return false;
    if (read_request_.current().valid || read_request_.next().valid) return false;
    if (write_request_.current().valid || write_request_.next().valid) return false;
    for (const auto& a : read_assembly_.current()) {
        if (a.active) return false;
    }
    return true;
}

size_t DRAMSim3Memory::in_flight_count() const {
    // A write is in flight from submit_write until its synthetic ack releases.
    size_t n = pending_write_acks_.current().size();
    for (const auto& a : read_assembly_.current()) {
        if (a.active) ++n;
    }
    return n;
}

void DRAMSim3Memory::reset() {
    // Phase 5 (sparkling-dazzling-starfish.md): reset_all() clears every
    // enrolled Reg's current_ AND next_. The cross-stage RegFifos
    // (request_fifo_, mem_responses_, mem_write_acks_) are owned by
    // TimingModel; resetting them is TimingModel's responsibility (or the
    // test fixture's, for fixtures that own local RegFifos).
    reset_all();
    // read_assembly_ defaults to an empty vector after reset_all(); re-size it.
    read_assembly_.initialize(std::vector<ReadAssembly>(cfg_.num_mshrs));
    phase_ = 0.0;
    fabric_cycle_ = 0;
    dram_ticks_ = 0;
    max_response_queue_ = 0;
    max_write_ack_queue_ = 0;
    rebuild_memory_system();
}

void DRAMSim3Memory::on_read_complete(uint64_t addr) {
    // Phase 5 (sparkling-dazzling-starfish.md): invoked from within
    // mem_->ClockTick() during evaluate_dram(). All state writes target
    // next_mut() on the DRAM-clock half's Reg containers; the response
    // push stages onto the TimingModel-owned mem_responses_ RegFifo.
    auto& chunk_to_mshr = read_chunk_to_mshr_.next_mut();
    auto it = chunk_to_mshr.find(addr);
    if (it == chunk_to_mshr.end()) return;
    const uint32_t mshr_id = it->second;
    chunk_to_mshr.erase(it);

    auto& assembly = read_assembly_.next_mut();
    auto& a = assembly[mshr_id];
    if (a.chunks_remaining == 0 || !a.active) return;
    --a.chunks_remaining;
    if (a.chunks_remaining == 0) {
        // Architectural invariant: the read response queue is bounded by
        // num_mshrs (at most one response per in-flight MSHR).
        assert(mem_responses_ &&
               "DRAMSim3Memory: mem_responses_ back-pointer required for completion");
        const std::size_t depth =
            mem_responses_->current_size() + mem_responses_->pushes_staged() + 1u;
        assert(depth <= read_response_queue_capacity_ &&
               "DRAMSim3Memory read response queue overflow on read completion");
        MemoryResponse resp{a.line_addr, mshr_id, false};
        mem_responses_->stage_push(resp);
        stats_.external_read_latency_total +=
            (fabric_cycle_ - a.submit_cycle);
        stats_.external_read_latency_count++;
        a.active = false;
        if (depth > max_response_queue_) {
            max_response_queue_ = depth;
        }
    }
}

} // namespace gpu_sim
