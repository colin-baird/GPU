#include "gpu_sim/timing/memory_interface.h"

namespace gpu_sim {

FixedLatencyMemory::FixedLatencyMemory(uint32_t latency, Stats& stats)
    : latency_(latency), stats_(stats) {
    // Phase 4 of current_mut() elimination (Pattern 3): the REGISTERED
    // request slots are PulseReg<PendingMemoryRequest>. Enrolled in
    // register_state so commit_all() / reset_all() drive them uniformly.
    // FixedLatencyMemory is NOT in TimingModel::tick()'s seed phase — the
    // default-to-T{} mechanism is PulseReg::commit()'s own internal
    // `next_ = T{}` reset after the current_ = next_ flip (see reg.h).
    // The previous shape relied on a mid-cycle current_mut().valid=false
    // clear by the consumer plus a tail-of-commit set_next(T{}) by the
    // owner; both are replaced by PulseReg's commit-time reset.
    register_state(&read_request_, &write_request_);
}

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
    auto& rr = read_request_.next_mut();
    rr.valid = true;
    rr.line_addr = line_addr;
    rr.mshr_id = mshr_id;
    // Stats increment at submit time (consistent with submit_read).
    stats_.external_memory_reads++;
}

void FixedLatencyMemory::set_next_write_request(uint32_t line_addr) {
    auto& wr = write_request_.next_mut();
    wr.valid = true;
    wr.line_addr = line_addr;
    stats_.external_memory_writes++;
}

void FixedLatencyMemory::evaluate() {
    // Memoryless-consumer pattern: read committed slot, drain into in_flight_.
    // No mid-cycle Q write — PulseReg's seed-to-T{} at the top of the next
    // tick will default the slot to invalid; the cache must explicitly stage
    // a new request via set_next_*_request to keep the slot live.
    if (read_request_.current().valid) {
        const auto& cr = read_request_.current();
        MemoryRequest req;
        req.line_addr = cr.line_addr;
        req.mshr_id = cr.mshr_id;
        req.is_write = false;
        req.cycles_remaining = latency_;
        in_flight_.push_back(req);
    }
    if (write_request_.current().valid) {
        const auto& cw = write_request_.current();
        MemoryRequest req;
        req.line_addr = cw.line_addr;
        req.mshr_id = 0;
        req.is_write = true;
        req.cycles_remaining = latency_;
        in_flight_.push_back(req);
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
    // Phase 4 of current_mut() elimination (Pattern 3): commit_all() flips
    // next_ -> current_ on the PulseReg<PendingMemoryRequest> slots, and
    // PulseReg::commit() also resets next_ to T{} after the flip — so no
    // tail set_next(T{}) is needed. A cycle on which the cache does not
    // re-stage a request latches T{} into current_ at the next commit.
    commit_all();
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
    // Phase 6 (reg.h migration): reset_all() clears both current_ AND next_
    // for every enrolled Reg — equivalent to today's pair of
    // `current_*_request_ = PendingMemoryRequest{}; next_*_request_ = ...`.
    reset_all();
}

} // namespace gpu_sim
