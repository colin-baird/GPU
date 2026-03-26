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

class ExternalMemoryInterface {
public:
    ExternalMemoryInterface(uint32_t latency, Stats& stats);

    void evaluate();
    void commit();
    void reset();

    bool submit_read(uint32_t line_addr, uint32_t mshr_id);
    bool submit_write(uint32_t line_addr);

    bool has_response() const { return !responses_.empty(); }
    MemoryResponse get_response();

private:
    uint32_t latency_;
    Stats& stats_;
    std::deque<MemoryRequest> in_flight_;
    std::deque<MemoryResponse> responses_;
};

} // namespace gpu_sim
