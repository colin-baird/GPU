#pragma once

#include "gpu_sim/types.h"
#include "gpu_sim/stats.h"
#include <deque>

namespace gpu_sim {

struct MemoryRequest {
    uint32_t line_addr;
    uint32_t mshr_id;
    bool is_write;
    uint32_t cycles_remaining;
};

struct MemoryResponse {
    uint32_t line_addr;
    uint32_t mshr_id;
    bool is_write;
};

// Abstract interface to external memory. The cache talks to this surface and
// is agnostic to the underlying timing model (fixed-latency, DRAMSim3, ...).
class ExternalMemoryInterface {
public:
    virtual ~ExternalMemoryInterface() = default;

    virtual void evaluate() = 0;
    virtual void commit() = 0;
    virtual void reset() = 0;

    virtual bool submit_read(uint32_t line_addr, uint32_t mshr_id) = 0;
    virtual bool submit_write(uint32_t line_addr) = 0;

    virtual bool has_response() const = 0;
    virtual MemoryResponse get_response() = 0;
    virtual bool is_idle() const = 0;
    virtual size_t in_flight_count() const = 0;
    virtual size_t response_count() const = 0;
};

// Original behavior: every request completes after exactly `latency` cycles.
class FixedLatencyMemory : public ExternalMemoryInterface {
public:
    FixedLatencyMemory(uint32_t latency, Stats& stats);

    void evaluate() override;
    void commit() override;
    void reset() override;

    bool submit_read(uint32_t line_addr, uint32_t mshr_id) override;
    bool submit_write(uint32_t line_addr) override;

    bool has_response() const override { return !responses_.empty(); }
    MemoryResponse get_response() override;
    bool is_idle() const override { return in_flight_.empty() && responses_.empty(); }
    size_t in_flight_count() const override { return in_flight_.size(); }
    size_t response_count() const override { return responses_.size(); }

private:
    uint32_t latency_;
    Stats& stats_;
    std::deque<MemoryRequest> in_flight_;
    std::deque<MemoryResponse> responses_;
};

} // namespace gpu_sim
