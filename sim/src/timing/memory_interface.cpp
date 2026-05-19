#include "gpu_sim/timing/memory_interface.h"

namespace gpu_sim {

FixedLatencyMemory::FixedLatencyMemory(uint32_t latency, Stats& stats)
    : latency_(latency), stats_(stats) {}

bool FixedLatencyMemory::submit_read(uint32_t line_addr, uint32_t mshr_id) {
    // Test-direct path: pushes into in_flight_ at submission time.
    // Production cache uses set_next_read_request + commit + evaluate
    // to model the REGISTERED 1-cycle admission.
    MemoryRequest req;
    req.line_addr = line_addr;
    req.mshr_id = mshr_id;
    req.is_write = false;
    req.cycles_remaining = latency_;
    in_flight_.push_back(req);
    stats_.external_memory_reads++;
    return true;
}

bool FixedLatencyMemory::submit_write(uint32_t line_addr) {
    MemoryRequest req;
    req.line_addr = line_addr;
    req.mshr_id = 0;
    req.is_write = true;
    req.cycles_remaining = latency_;
    in_flight_.push_back(req);
    stats_.external_memory_writes++;
    return true;
}

void FixedLatencyMemory::set_next_read_request(uint32_t line_addr, uint32_t mshr_id) {
    next_read_request_.valid = true;
    next_read_request_.line_addr = line_addr;
    next_read_request_.mshr_id = mshr_id;
    // Stats increment at submit time (consistent with submit_read).
    stats_.external_memory_reads++;
}

void FixedLatencyMemory::set_next_write_request(uint32_t line_addr) {
    next_write_request_.valid = true;
    next_write_request_.line_addr = line_addr;
    stats_.external_memory_writes++;
}

void FixedLatencyMemory::evaluate() {
    if (current_read_request_.valid) {
        MemoryRequest req;
        req.line_addr = current_read_request_.line_addr;
        req.mshr_id = current_read_request_.mshr_id;
        req.is_write = false;
        req.cycles_remaining = latency_;
        in_flight_.push_back(req);
        current_read_request_.valid = false;
    }
    if (current_write_request_.valid) {
        MemoryRequest req;
        req.line_addr = current_write_request_.line_addr;
        req.mshr_id = 0;
        req.is_write = true;
        req.cycles_remaining = latency_;
        in_flight_.push_back(req);
        current_write_request_.valid = false;
    }

    for (auto& req : in_flight_) {
        req.cycles_remaining--;
    }

    while (!in_flight_.empty() && in_flight_.front().cycles_remaining == 0) {
        auto& req = in_flight_.front();
        MemoryResponse resp;
        resp.line_addr = req.line_addr;
        resp.mshr_id = req.mshr_id;
        resp.is_write = req.is_write;
        // Reads complete onto the response queue; writes complete onto the
        // separate write-ack channel the cache drains unconditionally.
        if (req.is_write) {
            write_acks_.push_back(resp);
        } else {
            stats_.external_read_latency_total += latency_;
            stats_.external_read_latency_count++;
            responses_.push_back(resp);
        }
        in_flight_.pop_front();
    }
}

void FixedLatencyMemory::commit() {
    // Phase M5: flip REGISTERED request slots.
    current_read_request_ = next_read_request_;
    next_read_request_ = PendingMemoryRequest{};
    current_write_request_ = next_write_request_;
    next_write_request_ = PendingMemoryRequest{};
}

MemoryResponse FixedLatencyMemory::get_response() {
    MemoryResponse resp = responses_.front();
    responses_.pop_front();
    return resp;
}

MemoryResponse FixedLatencyMemory::get_write_ack() {
    MemoryResponse resp = write_acks_.front();
    write_acks_.pop_front();
    return resp;
}

void FixedLatencyMemory::reset() {
    in_flight_.clear();
    responses_.clear();
    write_acks_.clear();
    current_read_request_ = PendingMemoryRequest{};
    next_read_request_ = PendingMemoryRequest{};
    current_write_request_ = PendingMemoryRequest{};
    next_write_request_ = PendingMemoryRequest{};
}

} // namespace gpu_sim
