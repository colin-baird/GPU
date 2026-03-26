#include "gpu_sim/timing/memory_interface.h"

namespace gpu_sim {

ExternalMemoryInterface::ExternalMemoryInterface(uint32_t latency, Stats& stats)
    : latency_(latency), stats_(stats) {}

bool ExternalMemoryInterface::submit_read(uint32_t line_addr, uint32_t mshr_id) {
    MemoryRequest req;
    req.line_addr = line_addr;
    req.mshr_id = mshr_id;
    req.is_write = false;
    req.cycles_remaining = latency_;
    in_flight_.push_back(req);
    stats_.external_memory_reads++;
    return true;
}

bool ExternalMemoryInterface::submit_write(uint32_t line_addr) {
    MemoryRequest req;
    req.line_addr = line_addr;
    req.mshr_id = 0;
    req.is_write = true;
    req.cycles_remaining = latency_;
    in_flight_.push_back(req);
    stats_.external_memory_writes++;
    return true;
}

void ExternalMemoryInterface::evaluate() {
    for (auto& req : in_flight_) {
        req.cycles_remaining--;
    }

    while (!in_flight_.empty() && in_flight_.front().cycles_remaining == 0) {
        auto& req = in_flight_.front();
        MemoryResponse resp;
        resp.line_addr = req.line_addr;
        resp.mshr_id = req.mshr_id;
        resp.is_write = req.is_write;
        responses_.push_back(resp);
        in_flight_.pop_front();
    }
}

void ExternalMemoryInterface::commit() {}

MemoryResponse ExternalMemoryInterface::get_response() {
    MemoryResponse resp = responses_.front();
    responses_.pop_front();
    return resp;
}

void ExternalMemoryInterface::reset() {
    in_flight_.clear();
    responses_.clear();
}

} // namespace gpu_sim
