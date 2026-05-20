#pragma once

#include "gpu_sim/types.h"
#include "gpu_sim/stats.h"
#include "gpu_sim/timing/reg.h"
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

// Phase M5: REGISTERED forward request slots.
struct PendingMemoryRequest {
    bool valid = false;
    uint32_t line_addr = 0;
    uint32_t mshr_id = 0;
};

// Abstract interface to external memory. The cache talks to this surface and
// is agnostic to the underlying timing model (fixed-latency, DRAMSim3, ...).
class ExternalMemoryInterface {
public:
    virtual ~ExternalMemoryInterface() = default;

    virtual void evaluate() = 0;
    virtual void commit() = 0;
    virtual void reset() = 0;

    // Phase M5: REGISTERED request submission. Cache writes via
    // set_next_*_request gated on !next_request_stall(); commit() flips
    // next_ → current_; evaluate() at the next cycle drains current_*
    // into in_flight_ and proceeds with normal latency progression. The
    // bool return is gone — acceptance is guaranteed when the stall is
    // clear at submission time.
    virtual void set_next_read_request(uint32_t line_addr, uint32_t mshr_id) = 0;
    virtual void set_next_write_request(uint32_t line_addr) = 0;
    // COMBINATIONAL backward stall — reads end-of-cycle queue/FIFO state.
    // Cache reads same-cycle (cache.evaluate runs before mem_if.evaluate
    // in tick order; cache reads the value mem_if produced last cycle).
    virtual bool next_request_stall() const = 0;

    // Direct synchronous API. Test-direct path: tests that exercise mem_if
    // in isolation (test_dramsim3_memory, test_timing_components) push
    // requests straight into in_flight_ via these calls. Production cache
    // uses set_next_*_request on miss / write-buffer drain paths.
    virtual bool submit_read(uint32_t line_addr, uint32_t mshr_id) = 0;
    virtual bool submit_write(uint32_t line_addr) = 0;

    // Phase M5: canonical accessor name. Cache.handle_responses runs at
    // the top of cache.evaluate — before mem_if.evaluate produces this
    // cycle's responses — so the read is already against committed state
    // from the end of the previous cycle. The next_has_response() name is
    // preserved as an alias for backward compatibility.
    virtual bool current_has_response() const = 0;
    bool next_has_response() const { return current_has_response(); }
    virtual MemoryResponse get_response() = 0;

    // Separate write-ack channel. External memory delivers write
    // completions here, NOT on the read-fill response queue, so the cache
    // can drain write acks unconditionally every cycle (the deadlock fix:
    // a write-ack pin would otherwise never clear when a fill is deferred).
    // REGISTERED forward edge, mirroring current_has_response/get_response.
    virtual bool current_has_write_ack() const = 0;
    virtual MemoryResponse get_write_ack() = 0;
    virtual size_t write_ack_count() const = 0;

    virtual bool is_idle() const = 0;
    virtual size_t in_flight_count() const = 0;
    virtual size_t response_count() const = 0;
};

// Original behavior: every request completes after exactly `latency` cycles.
//
// Phase 4 of current_mut() elimination (Pattern 3): the REGISTERED request
// slots are wrapped as PulseReg<PendingMemoryRequest>. PulseReg::commit()
// flips current_ = next_ and ALSO resets next_ to T{} after the flip, so a
// cycle on which the cache does not stage a fresh request latches T{}
// into current_ at the next commit. evaluate() reads current() (last
// cycle's request, if any) and drains it into in_flight_; no mid-cycle Q
// write. The previous shape was Reg<T> with a mid-cycle
// current_mut().valid=false clear plus a tail-of-commit set_next(T{})
// reset; both are replaced by PulseReg's commit-time reset.
class FixedLatencyMemory : public ExternalMemoryInterface, public RegisteredStage {
public:
    FixedLatencyMemory(uint32_t latency, Stats& stats);

    void evaluate() override;
    void commit() override;
    void reset() override;

    // Phase M5: REGISTERED forward request path.
    void set_next_read_request(uint32_t line_addr, uint32_t mshr_id) override;
    void set_next_write_request(uint32_t line_addr) override;
    bool next_request_stall() const override { return false; } // unbounded queue

    bool submit_read(uint32_t line_addr, uint32_t mshr_id) override;
    bool submit_write(uint32_t line_addr) override;

    bool current_has_response() const override { return !responses_.empty(); }
    MemoryResponse get_response() override;
    bool current_has_write_ack() const override { return !write_acks_.empty(); }
    MemoryResponse get_write_ack() override;
    size_t write_ack_count() const override { return write_acks_.size(); }
    bool is_idle() const override {
        return in_flight_.empty() && responses_.empty() && write_acks_.empty()
               && !read_request_.current().valid && !read_request_.next().valid
               && !write_request_.current().valid && !write_request_.next().valid;
    }
    size_t in_flight_count() const override { return in_flight_.size(); }
    size_t response_count() const override { return responses_.size(); }

private:
    uint32_t latency_;  // config
    Stats& stats_;      // config (back-pointer)
    // deliberate-non-register: internal scheduling state. evaluate() pushes
    // and pops these in a single pass each cycle and the per-entry
    // cycles_remaining counter is decremented in place. Wrapping as
    // Reg<std::deque<T>> would mean a whole-deque seed copy at top of tick,
    // an evaluate that mutates next_, and commit_all() that flips. responses_
    // and write_acks_ are popped CROSS-CLASS by the cache (cache.evaluate
    // calls get_response() / get_write_ack() during its own evaluate, before
    // mem_if.evaluate runs) — staged-pop semantics would require redesigning
    // that cross-stage handshake into a Wire-mediated pull. The Phase 6
    // audit flagged conversion; deferred pending the handshake redesign.
    std::deque<MemoryRequest> in_flight_;
    std::deque<MemoryResponse> responses_;
    std::deque<MemoryResponse> write_acks_;
    // Phase 4 of current_mut() elimination (Pattern 3): PulseReg<T> request
    // slots. Default to T{} each cycle via PulseReg::commit()'s post-flip
    // reset; cache overrides by calling set_next_*_request during its
    // evaluate.
    PulseReg<PendingMemoryRequest> read_request_;
    PulseReg<PendingMemoryRequest> write_request_;
};

} // namespace gpu_sim
