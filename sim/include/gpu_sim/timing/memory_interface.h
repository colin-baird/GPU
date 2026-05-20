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
//
// Phase 4 (close-the-Reg-family-migration): the response / write-ack queues
// are cross-stage FIFOs touched by both the memory backend (producer) and the
// cache (consumer). The cross-stage-FIFO ownership pattern established in
// Phase 3 applies: a TimingModel-owned RegFifo<MemoryResponse> committed in
// the dedicated ungated cross-stage pass. The interface accessors below
// model the RTL-faithful current()/stage_pop() shape: the cache reads
// current_response_front() (last cycle's committed completion, the natural
// one-cycle FIFO latency a real hardware FIFO presents) and calls
// stage_response_pop() to claim it; the staged pop applies at the cross-
// stage FIFO commit. set_response_queues() wires the TimingModel-owned
// RegFifos into the backend so the backend's evaluate() can stage_push().
//
// The FixedLatencyMemory backend implements this contract end-to-end.
// DRAMSim3Memory continues to route through its internal std::deque<>
// completions for byte-identity in Phase 4 — its stage_response_pop()
// performs an immediate pop on the internal deque. Phase 5 will move the
// DRAMSim3 completion path onto the TimingModel-owned RegFifos as well,
// at which point that one-cycle CDC latency lands as a documented delta.
class ExternalMemoryInterface {
public:
    virtual ~ExternalMemoryInterface() = default;

    // Phase 5 (sparkling-dazzling-starfish.md): evaluate is split into a
    // fabric-clock half and a DRAM-clock half so the DRAMSim3 backend can
    // sequence them at independent positions in TimingModel::tick()'s
    // sweep, surfacing the natural one-fabric-cycle CDC traversal latency
    // through the TimingModel-owned request_fifo_ that the prior
    // submit-then-drain in one body collapsed. FixedLatencyMemory does all
    // its work in evaluate_fabric() and overrides evaluate_dram() as a
    // no-op (single-clock backend; no CDC).
    virtual void evaluate_fabric() = 0;
    virtual void evaluate_dram() = 0;
    virtual void commit() = 0;
    virtual void reset() = 0;

    // Convenience wrapper for test fixtures / call sites that drive the
    // backend manually (no TimingModel sweep). Runs both halves
    // back-to-back, then the caller is responsible for committing any
    // cross-stage FIFO it owns (for DRAMSim3 fixtures, the local
    // RegFifo<DRAMSim3PendingChunk>) so the DRAM-side reads of
    // request_fifo_.current() see the fabric-side pushes one tick later
    // — same CDC discipline TimingModel::tick() enforces.
    void evaluate() { evaluate_fabric(); evaluate_dram(); }

    // Phase 4 (close-the-Reg-family-migration): cross-stage response FIFO
    // back-pointers. Set once at TimingModel construction (FixedLatencyMemory
    // wires its evaluate-time stage_push() onto these; DRAMSim3 currently
    // ignores the back-pointers and keeps its internal deques — Phase 5 will
    // adopt the same pattern). Both pointers may be nullptr for unit-test
    // fixtures that exercise the backend in isolation (those tests own local
    // RegFifos and commit them directly, mirroring the Phase-3 addr-gen-fifo
    // fixture pattern).
    virtual void set_response_queues(RegFifo<MemoryResponse>* responses,
                                     RegFifo<MemoryResponse>* write_acks) = 0;

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

    // Phase 4 (close-the-Reg-family-migration): peek + staged-pop on the
    // cross-stage response FIFO. The cache calls these during its own
    // evaluate() — earlier in the tick sweep than mem_if.evaluate() runs —
    // so the read is against committed state from the end of the previous
    // cycle. stage_response_pop() stages the pop; it applies at the cross-
    // stage FIFO commit. next_has_response() is preserved as an alias for
    // backward compatibility.
    virtual bool current_has_response() const = 0;
    bool next_has_response() const { return current_has_response(); }
    virtual const MemoryResponse& current_response_front() const = 0;
    virtual void stage_response_pop() = 0;

    // Separate write-ack channel. External memory delivers write
    // completions here, NOT on the read-fill response queue, so the cache
    // can drain write acks unconditionally every cycle (the deadlock fix:
    // a write-ack pin would otherwise never clear when a fill is deferred).
    // Same cross-stage discipline as the response channel.
    virtual bool current_has_write_ack() const = 0;
    virtual const MemoryResponse& current_write_ack_front() const = 0;
    virtual void stage_write_ack_pop() = 0;
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
//
// Phase 4 (close-the-Reg-family-migration): the response / write-ack queues
// are cross-stage RegFifo<MemoryResponse> owned by TimingModel (peer of the
// cache, not a member of mem_if), wired via set_response_queues() and
// committed in TimingModel::commit_cross_stage_fifos(). in_flight_ is
// internal to FixedLatencyMemory (no cross-class consumer) and is wrapped
// as Reg<std::deque<MemoryRequest>> enrolled in mem_if's own RegisteredStage.
// FixedLatencyMemory is never gated, so no seed_next() is required: the
// mutation surface is `in_flight_.next_mut()` and commit() flips at the
// cycle boundary (next cycle's evaluate sees the committed survivors).
class FixedLatencyMemory : public ExternalMemoryInterface, public RegisteredStage {
public:
    FixedLatencyMemory(uint32_t latency, Stats& stats);

    // Phase 5 (sparkling-dazzling-starfish.md): single-clock backend — all
    // work happens in evaluate_fabric(); evaluate_dram() is a no-op. Keeps
    // the surface uniform across backends.
    void evaluate_fabric() override;
    void evaluate_dram() override {}
    void commit() override;
    void reset() override;

    // Phase 4 (close-the-Reg-family-migration): wire the TimingModel-owned
    // cross-stage response / write-ack RegFifos. Either pointer may be
    // nullptr in unit-test fixtures (those own a local RegFifo and commit
    // it explicitly, mirroring the Phase-3 addr-gen-fifo pattern).
    void set_response_queues(RegFifo<MemoryResponse>* responses,
                             RegFifo<MemoryResponse>* write_acks) override {
        mem_responses_ = responses;
        mem_write_acks_ = write_acks;
    }

    // Phase M5: REGISTERED forward request path.
    void set_next_read_request(uint32_t line_addr, uint32_t mshr_id) override;
    void set_next_write_request(uint32_t line_addr) override;
    bool next_request_stall() const override { return false; } // unbounded queue

    bool submit_read(uint32_t line_addr, uint32_t mshr_id) override;
    bool submit_write(uint32_t line_addr) override;

    bool current_has_response() const override {
        return mem_responses_ && !mem_responses_->current_empty();
    }
    const MemoryResponse& current_response_front() const override {
        return mem_responses_->current_front();
    }
    void stage_response_pop() override {
        if (mem_responses_) mem_responses_->stage_pop();
    }
    bool current_has_write_ack() const override {
        return mem_write_acks_ && !mem_write_acks_->current_empty();
    }
    const MemoryResponse& current_write_ack_front() const override {
        return mem_write_acks_->current_front();
    }
    void stage_write_ack_pop() override {
        if (mem_write_acks_) mem_write_acks_->stage_pop();
    }
    size_t write_ack_count() const override {
        return mem_write_acks_ ? mem_write_acks_->current_size() : 0u;
    }
    bool is_idle() const override {
        return in_flight_.current().empty()
               && (!mem_responses_ || mem_responses_->current_empty())
               && (!mem_write_acks_ || mem_write_acks_->current_empty())
               && !read_request_.current().valid && !read_request_.next().valid
               && !write_request_.current().valid && !write_request_.next().valid;
    }
    size_t in_flight_count() const override { return in_flight_.current().size(); }
    size_t response_count() const override {
        return mem_responses_ ? mem_responses_->current_size() : 0u;
    }

private:
    uint32_t latency_;  // config
    Stats& stats_;      // config (back-pointer)
    // Phase 4 (close-the-Reg-family-migration): internal in-flight queue.
    // Mutated only by FixedLatencyMemory; no cross-class consumer. Wrapped
    // as Reg<std::deque<T>> and enrolled in this stage's RegisteredStage
    // (ungated — FixedLatencyMemory has no writeback-stall gate). evaluate()
    // mutates in_flight_.next_mut() in place (decrement cycles_remaining,
    // pop completions, push from current_*_request_). commit_all() flips
    // next_ -> current_; subsequent cycles' evaluate sees the committed
    // survivors (since Reg::commit sets current_=next_, the next cycle's
    // mutation begins from the post-commit state, no seed needed).
    Reg<std::deque<MemoryRequest>> in_flight_;
    // Phase 4 (close-the-Reg-family-migration): back-pointers to the
    // TimingModel-owned cross-stage response / write-ack RegFifos. The
    // FIFOs are peers of FixedLatencyMemory and L1Cache (committed in
    // TimingModel::commit_cross_stage_fifos()), not members of either,
    // borrowing the WarpState[] ownership pattern. nullptr-tolerant for
    // unit-test fixtures that own a local RegFifo and commit it directly.
    RegFifo<MemoryResponse>* mem_responses_ = nullptr;  // timing-naming-allow: back-pointer to TimingModel-owned cross-stage RegFifo (Phase-3 pattern); the FIFO itself is committed at TimingModel::commit_cross_stage_fifos().
    RegFifo<MemoryResponse>* mem_write_acks_ = nullptr; // timing-naming-allow: back-pointer to TimingModel-owned cross-stage RegFifo (Phase-3 pattern); the FIFO itself is committed at TimingModel::commit_cross_stage_fifos().
    // Phase 4 of current_mut() elimination (Pattern 3): PulseReg<T> request
    // slots. Default to T{} each cycle via PulseReg::commit()'s post-flip
    // reset; cache overrides by calling set_next_*_request during its
    // evaluate.
    PulseReg<PendingMemoryRequest> read_request_;
    PulseReg<PendingMemoryRequest> write_request_;
};

} // namespace gpu_sim
