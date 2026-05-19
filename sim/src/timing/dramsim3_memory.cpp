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
      ticks_per_fabric_(cfg.dram_clock_mhz / cfg.fpga_clock_mhz),
      read_assembly_(cfg.num_mshrs) {
    if (cfg.dramsim3_config_path.empty()) {
        throw std::invalid_argument(
            "DRAMSim3Memory requires a non-empty dramsim3_config_path");
    }
    if (chunks_per_line_ == 0) {
        throw std::invalid_argument(
            "DRAMSim3Memory: cache_line_size_bytes < dramsim3_bytes_per_burst");
    }
    // Phase 6 (reg.h migration): enroll the REGISTERED request slots so
    // commit_all() / reset_all() drive them uniformly. The backend
    // intentionally has no seed_next() and is not in TimingModel::tick()'s
    // seed phase — read_request_ / write_request_ are memoryless-consumer
    // slots (evaluate consumes whatever sits in current() and clears it via
    // current_mut(); auto-seeding next from current would re-latch the
    // consumed request). See memory_interface.h for the full discipline.
    register_state(&read_request_, &write_request_);
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
    if (mshr_id >= read_assembly_.size()) {
        throw std::out_of_range("DRAMSim3Memory: mshr_id out of range");
    }
    // Architectural invariant: at most num_mshrs reads can be in flight at
    // once, and the request FIFO reserves num_mshrs * chunks_per_line slots
    // for reads (= request_fifo_depth_ - write_region_capacity_). Cache
    // miss paths reach this via set_next_read_request → evaluate(), which
    // discards the bool, so an overflow would silently drop a read and
    // leave the MSHR allocated forever. The assert converts that into an
    // immediate failure.
    assert(request_fifo_.size() + chunks_per_line_ <= request_fifo_depth_ &&
           "DRAMSim3Memory request FIFO would overflow on submit_read; "
           "writes must be confined to the write region (submit_write enforces "
           "this) so reads always have their reserved slots");
    if (request_fifo_.size() + chunks_per_line_ > request_fifo_depth_) {
        return false;
    }
    auto& a = read_assembly_[mshr_id];
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
        request_fifo_.push_back(
            {base + static_cast<uint64_t>(i) * bytes_per_burst_,
             line_addr, mshr_id, false});
    }
    stats_.external_memory_reads++;
    return true;
}

bool DRAMSim3Memory::submit_write(uint32_t line_addr) {
    // Writes are confined to the write region of the FIFO (= wb_depth *
    // chunks_per_line slots). Beyond that, return false — the cache must
    // hold the entry in its write_buffer and retry next cycle. This is the
    // architectural backpressure from mem_if to the cache; without it, the
    // FIFO would grow unbounded under sustained write traffic and either
    // overflow (silent drop) or starve reads of their reserved slots.
    if (write_chunks_in_fifo_ + chunks_per_line_ > write_region_capacity_) {
        return false;
    }
    const uint64_t base = static_cast<uint64_t>(line_addr) * line_size_;
    for (uint32_t i = 0; i < chunks_per_line_; ++i) {
        request_fifo_.push_back(
            {base + static_cast<uint64_t>(i) * bytes_per_burst_,
             line_addr, 0, true});
    }
    write_chunks_in_fifo_ += chunks_per_line_;
    stats_.external_memory_writes++;
    // Schedule one synthetic write ack for this write-through, released
    // write_commit_latency_tck_ DRAM cycles from now (the write being issued
    // to DRAM). Exactly one ack per submit_write — 1:1 with the cache's
    // outstanding-write counter, with no same-line folding.
    pending_write_acks_.push_back(
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
    // Stall when the write region of the request FIFO can't accept another
    // line's worth of chunks. Reads have their own reserved region and
    // are never stalled (the architectural invariant in the file's class
    // doc above guarantees this). Conservative for the cache's purposes:
    // stall any cmd when writes can't drain.
    return write_chunks_in_fifo_ + chunks_per_line_ > write_region_capacity_;
}

void DRAMSim3Memory::evaluate() {
    // Phase M5: drain current_*_request_ into the request FIFO at top of
    // evaluate. Memoryless-consumer: consume the committed slot and
    // invalidate it via the documented current_mut() escape hatch
    // (matches today's pre-migration `current_*_request_.valid = false`).
    if (read_request_.current().valid) {
        submit_read(read_request_.current().line_addr,
                    read_request_.current().mshr_id);
        read_request_.current_mut().valid = false;
    }
    if (write_request_.current().valid) {
        submit_write(write_request_.current().line_addr);
        write_request_.current_mut().valid = false;
    }

    ++fabric_cycle_;
    phase_ += ticks_per_fabric_;
    while (phase_ >= 1.0) {
        if (!request_fifo_.empty()) {
            const auto& c = request_fifo_.front();
            if (mem_->WillAcceptTransaction(c.chunk_byte_addr, c.is_write)) {
                if (c.is_write) {
                    --write_chunks_in_fifo_;
                } else {
                    read_chunk_to_mshr_[c.chunk_byte_addr] = c.mshr_id;
                }
                mem_->AddTransaction(c.chunk_byte_addr, c.is_write);
                request_fifo_.pop_front();
            }
        }
        mem_->ClockTick();
        ++dram_ticks_;
        phase_ -= 1.0;
    }

    // Release synthetic write acks whose commit latency has elapsed.
    // pending_write_acks_ is sorted by release tick (submit order, constant
    // latency), so a front-drain is correct.
    while (!pending_write_acks_.empty() &&
           pending_write_acks_.front().release_dram_tick <= dram_ticks_) {
        assert(write_acks_.size() < write_ack_queue_capacity_ &&
               "DRAMSim3Memory write-ack queue overflow on synthetic release");
        write_acks_.push_back({pending_write_acks_.front().line_addr, 0, true});
        pending_write_acks_.pop_front();
        if (write_acks_.size() > max_write_ack_queue_) {
            max_write_ack_queue_ = write_acks_.size();
        }
    }
}

void DRAMSim3Memory::commit() {
    // Phase 6 (reg.h migration): flip the REGISTERED request slots via
    // commit_all(), then explicitly clear the staged slot — equivalent to
    // today's `next_*_request_ = PendingMemoryRequest{}` at the tail of
    // commit(). See memory_interface.cpp for the full memoryless-consumer
    // rationale.
    commit_all();
    read_request_.set_next(PendingMemoryRequest{});
    write_request_.set_next(PendingMemoryRequest{});
}

MemoryResponse DRAMSim3Memory::get_response() {
    MemoryResponse r = responses_.front();
    responses_.pop_front();
    return r;
}

MemoryResponse DRAMSim3Memory::get_write_ack() {
    MemoryResponse r = write_acks_.front();
    write_acks_.pop_front();
    return r;
}

bool DRAMSim3Memory::is_idle() const {
    if (!request_fifo_.empty() || !responses_.empty() || !write_acks_.empty()) return false;
    if (!pending_write_acks_.empty()) return false;
    if (read_request_.current().valid || read_request_.next().valid) return false;
    if (write_request_.current().valid || write_request_.next().valid) return false;
    for (const auto& a : read_assembly_) {
        if (a.active) return false;
    }
    return true;
}

size_t DRAMSim3Memory::in_flight_count() const {
    // A write is in flight from submit_write until its synthetic ack releases.
    size_t n = pending_write_acks_.size();
    for (const auto& a : read_assembly_) {
        if (a.active) ++n;
    }
    return n;
}

void DRAMSim3Memory::reset() {
    request_fifo_.clear();
    write_chunks_in_fifo_ = 0;
    responses_.clear();
    write_acks_.clear();
    pending_write_acks_.clear();
    for (auto& a : read_assembly_) a = {};
    read_chunk_to_mshr_.clear();
    // Phase 6 (reg.h migration): reset_all() clears both current_ AND next_
    // for every enrolled Reg — equivalent to today's
    // `current_*_request_ = PendingMemoryRequest{}; next_*_request_ = ...`.
    reset_all();
    phase_ = 0.0;
    fabric_cycle_ = 0;
    dram_ticks_ = 0;
    max_response_queue_ = 0;
    max_write_ack_queue_ = 0;
    rebuild_memory_system();
}

void DRAMSim3Memory::on_read_complete(uint64_t addr) {
    auto it = read_chunk_to_mshr_.find(addr);
    if (it == read_chunk_to_mshr_.end()) return;
    const uint32_t mshr_id = it->second;
    read_chunk_to_mshr_.erase(it);

    auto& a = read_assembly_[mshr_id];
    if (a.chunks_remaining == 0 || !a.active) return;
    --a.chunks_remaining;
    if (a.chunks_remaining == 0) {
        // Architectural invariant: the read response queue is bounded by
        // num_mshrs (at most one response per in-flight MSHR). Violation
        // means either the cache is producing more in-flight reads than its
        // own resources should permit, or the bound derivation is wrong.
        // Either way, silently growing the queue would mask a real bug.
        assert(responses_.size() < read_response_queue_capacity_ &&
               "DRAMSim3Memory read response queue overflow on read completion");
        responses_.push_back({a.line_addr, mshr_id, false});
        stats_.external_read_latency_total +=
            (fabric_cycle_ - a.submit_cycle);
        stats_.external_read_latency_count++;
        a.active = false;
        if (responses_.size() > max_response_queue_) {
            max_response_queue_ = responses_.size();
        }
    }
}

} // namespace gpu_sim
