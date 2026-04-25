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
      response_queue_capacity_(cfg.num_mshrs + cfg.write_buffer_depth +
                               (cfg.cache_line_size_bytes /
                                cfg.dramsim3_bytes_per_burst)),
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
    auto wcb = [this](uint64_t a) { on_write_complete(a); };
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
    // for reads (= request_fifo_depth_ - write_region_capacity_). The cache
    // ignores the bool return on submit_read (cache.cpp:112,190), so an
    // overflow here would silently drop a read and leave the MSHR allocated
    // forever. The assert converts that into an immediate failure.
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
    auto& w = write_assembly_[line_addr];
    w.chunks_remaining = static_cast<uint16_t>(w.chunks_remaining + chunks_per_line_);

    const uint64_t base = static_cast<uint64_t>(line_addr) * line_size_;
    for (uint32_t i = 0; i < chunks_per_line_; ++i) {
        request_fifo_.push_back(
            {base + static_cast<uint64_t>(i) * bytes_per_burst_,
             line_addr, 0, true});
    }
    write_chunks_in_fifo_ += chunks_per_line_;
    stats_.external_memory_writes++;
    return true;
}

void DRAMSim3Memory::evaluate() {
    phase_ += ticks_per_fabric_;
    while (phase_ >= 1.0) {
        if (!request_fifo_.empty()) {
            const auto& c = request_fifo_.front();
            if (mem_->WillAcceptTransaction(c.chunk_byte_addr, c.is_write)) {
                if (c.is_write) {
                    write_chunk_to_line_[c.chunk_byte_addr] = c.line_addr;
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
}

void DRAMSim3Memory::commit() {}

MemoryResponse DRAMSim3Memory::get_response() {
    MemoryResponse r = responses_.front();
    responses_.pop_front();
    return r;
}

bool DRAMSim3Memory::is_idle() const {
    if (!request_fifo_.empty() || !responses_.empty()) return false;
    if (!write_assembly_.empty()) return false;
    for (const auto& a : read_assembly_) {
        if (a.active) return false;
    }
    return true;
}

size_t DRAMSim3Memory::in_flight_count() const {
    size_t n = write_assembly_.size();
    for (const auto& a : read_assembly_) {
        if (a.active) ++n;
    }
    return n;
}

void DRAMSim3Memory::reset() {
    request_fifo_.clear();
    write_chunks_in_fifo_ = 0;
    responses_.clear();
    for (auto& a : read_assembly_) a = {};
    write_assembly_.clear();
    read_chunk_to_mshr_.clear();
    write_chunk_to_line_.clear();
    phase_ = 0.0;
    dram_ticks_ = 0;
    max_response_queue_ = 0;
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
        // Architectural invariant: the response queue is bounded by
        // num_mshrs + write_buffer_depth + chunks_per_line. Violation means
        // either the cache is producing more in-flight transactions than its
        // own resources should permit, or the bound derivation is wrong.
        // Either way, silently growing the queue would mask a real bug.
        assert(responses_.size() < response_queue_capacity_ &&
               "DRAMSim3Memory response queue overflow on read completion");
        responses_.push_back({a.line_addr, mshr_id, false});
        a.active = false;
        if (responses_.size() > max_response_queue_) {
            max_response_queue_ = responses_.size();
        }
    }
}

void DRAMSim3Memory::on_write_complete(uint64_t addr) {
    auto cit = write_chunk_to_line_.find(addr);
    if (cit == write_chunk_to_line_.end()) return;
    const uint32_t line_addr = cit->second;
    write_chunk_to_line_.erase(cit);

    auto wit = write_assembly_.find(line_addr);
    if (wit == write_assembly_.end()) return;
    auto& w = wit->second;
    if (w.chunks_remaining == 0) return;
    --w.chunks_remaining;
    if (w.chunks_remaining == 0) {
        assert(responses_.size() < response_queue_capacity_ &&
               "DRAMSim3Memory response queue overflow on write completion");
        responses_.push_back({line_addr, 0, true});
        write_assembly_.erase(wit);
        if (responses_.size() > max_response_queue_) {
            max_response_queue_ = responses_.size();
        }
    }
}

} // namespace gpu_sim
