#pragma once

#include "gpu_sim/types.h"
#include "gpu_sim/timing/mshr.h"
#include "gpu_sim/timing/memory_interface.h"
#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/timing/load_gather_buffer.h"
#include "gpu_sim/timing/reg.h"
#include "gpu_sim/stats.h"
#include <vector>
#include <deque>
#include <optional>

namespace gpu_sim {

struct CacheTag {
    bool valid = false;
    uint32_t tag = 0;
    // The "chain pin": set when a primary fill installs a line and has a
    // non-empty dependent chain; cleared when the last secondary drains.
    // This is one of two pin reasons combined by L1Cache::is_pinned() — the
    // other is the per-set outstanding-write counter (write-ack pin). A
    // different-tag miss into a pinned set (either reason) stalls LINE_PINNED.
    bool pinned = false;
};

struct PendingCacheFill {
    bool valid = false;
    MemoryResponse response{};
};

enum class CacheStallReason {
    NONE,
    MSHR_FULL,
    WRITE_BUFFER_FULL,
    LINE_PINNED
};

struct CacheMissTraceEvent {
    bool valid = false;
    uint32_t warp_id = 0;
    uint32_t line_addr = 0;
    bool is_store = false;
    uint32_t pc = 0;
    uint32_t raw_instruction = 0;
    // True when this miss was allocated as a secondary MSHR (same-line merge).
    bool merged_secondary = false;
};

struct CacheFillTraceEvent {
    bool valid = false;
    uint32_t warp_id = 0;
    uint32_t line_addr = 0;
    bool is_store = false;
    uint32_t pc = 0;
    uint32_t raw_instruction = 0;
    // Length of the dependent chain (primary + secondaries) at fill time.
    uint32_t chain_length_at_fill = 0;
    // True when complete_fill deferred because the target set was pinned by a
    // different line. The fill stays in pending_fill_ and is retried next cycle.
    bool deferred = false;
};

struct CacheSecondaryDrainTraceEvent {
    bool valid = false;
    uint32_t warp_id = 0;
    uint32_t line_addr = 0;
    bool is_store = false;
    uint32_t pc = 0;
    uint32_t raw_instruction = 0;
};

struct CachePinStallTraceEvent {
    bool valid = false;
    uint32_t warp_id = 0;
    uint32_t requested_line_addr = 0;
    uint32_t pinned_line_addr = 0;
    bool is_store = false;
};

// Phase M3: REGISTERED forward command path between coalescing and cache.
struct LoadCommand {
    bool valid = false;
    uint32_t addr = 0;
    uint32_t warp_id = 0;
    uint32_t lane_mask = 0;
    std::array<uint32_t, WARP_SIZE> results{};
    uint64_t issue_cycle = 0;
    uint32_t pc = 0;
    uint32_t raw_instruction = 0;
};

struct StoreCommand {
    bool valid = false;
    uint32_t line_addr = 0;
    uint32_t warp_id = 0;
    uint64_t issue_cycle = 0;
    uint32_t pc = 0;
    uint32_t raw_instruction = 0;
};

class L1Cache : public RegisteredStage {
public:
    L1Cache(uint32_t cache_size, uint32_t line_size, uint32_t num_mshrs,
            uint32_t write_buffer_depth, uint32_t max_outstanding_writes,
            ExternalMemoryInterface& mem_if,
            LoadGatherBufferFile& gather_file, Stats& stats);

    // Direct synchronous API. Used by tests and called internally by
    // L1Cache::evaluate to process the REGISTERED current load/store cmd
    // slots. Returns true when the request was accepted; tests rely on this
    // bool.
    bool process_load(uint32_t addr, uint32_t warp_id, uint32_t lane_mask,
                      const std::array<uint32_t, WARP_SIZE>& results,
                      uint64_t issue_cycle, uint32_t pc, uint32_t raw_instruction);
    bool process_store(uint32_t line_addr, uint32_t warp_id, uint64_t issue_cycle,
                       uint32_t pc, uint32_t raw_instruction);

    // Phase M3 (valid/ready): REGISTERED command setters. Coalescing writes
    // the staged slot; commit() flips it into the committed slot; evaluate()
    // processes the cmd this cycle and unconditionally clears it (cache is
    // memoryless — no retry state). Whether processing succeeded is exposed
    // via next_cmd_ready(); on failure (port lost to FILL/secondary, MSHR
    // full, pin/wb conflict) the cmd is dropped and coalescing must
    // re-stage from its own retry state (current_entry_/serial_index_).
    void set_next_load_cmd(uint32_t addr, uint32_t warp_id, uint32_t lane_mask,
                           const std::array<uint32_t, WARP_SIZE>& results,
                           uint64_t issue_cycle, uint32_t pc, uint32_t raw_instruction);
    void set_next_store_cmd(uint32_t line_addr, uint32_t warp_id, uint64_t issue_cycle,
                            uint32_t pc, uint32_t raw_instruction);

    // Phase M3 (valid/ready): the consumer-side ready signal. True iff
    // this cycle's cache.evaluate processed (and accepted) a cmd from the
    // committed slot. Tick order has cache.evaluate before coalescing.evaluate
    // so the producer reads ready combinationally after cache has set it.
    // The throughput guarantee "1 cmd/cycle when ready" is enforced by an
    // assert in evaluate. Phase 7: backed by Wire<bool> — one-line forwarder.
    bool next_cmd_ready() const { return next_cmd_ready_.value(); }

    // Generic resource-exhaustion accessor for trace classification only.
    // Returns the structural reason the LDST FIFO head warp is waiting on
    // the cache (MSHR_FULL / WRITE_BUFFER_FULL / LINE_PINNED) regardless
    // of any specific in-flight cmd. NONE when no resources are exhausted.
    // Order: MSHR_FULL > WRITE_BUFFER_FULL > LINE_PINNED > NONE.
    CacheStallReason next_cmd_stall_reason() const;

    // Handle one pending memory response (MSHR fill) per cycle. Load fills
    // deposit lane data into the owning warp's gather buffer; store fills push
    // the line into the write buffer.
    void handle_responses();

    // Drain write buffer
    void drain_write_buffer();

    void evaluate();
    void commit();
    void reset();

    // Phase 9 cache external boundary: each accessor declares its
    // discipline class. See resources/timing_discipline.md.
    //
    // COMBINATIONAL same-tick (single-slot, direct mutation, reset at top
    // of evaluate). Coalescing reads next_stalled() / next_stall_reason()
    // during its own evaluate, after cache.evaluate has already produced
    // this cycle's stall outcome. Models a same-cycle backpressure handshake
    // where the cache's stall signal is combinationally driven from
    // registered tag / write-buffer / pending_fill state.
    // Phase 7: backed by Wire<bool> / Wire<CacheStallReason> — one-line
    // forwarders to wire_.value().
    bool next_stalled() const { return stalled_.value(); }
    CacheStallReason next_stall_reason() const { return stall_reason_.value(); }
    //
    // REGISTERED (next/current pair flipped by commit()). Trace recording
    // runs at end-of-tick after cache.commit(); reads return committed
    // state. pending_fill_ also persists across cycles as a deferred-fill
    // carrier.
    bool is_idle() const;
    uint32_t active_mshr_count() const;
    std::vector<uint32_t> active_mshr_warps() const;
    size_t write_buffer_size() const { return write_buffer_.current_size(); }
    const PendingCacheFill& current_pending_fill() const { return pending_fill_.current(); }
    const MSHRFile& mshrs() const { return mshrs_; }
    const CacheMissTraceEvent& current_last_miss_event() const { return last_miss_event_.current(); }
    const CacheFillTraceEvent& current_last_fill_event() const { return last_fill_event_.current(); }
    const CacheSecondaryDrainTraceEvent& current_last_drain_event() const { return last_drain_event_.current(); }
    const CachePinStallTraceEvent& current_last_pin_stall_event() const { return last_pin_stall_event_.current(); }
    uint32_t pinned_line_count() const;
    uint32_t secondary_mshr_count() const;

private:
    bool complete_fill(const MemoryResponse& resp);
    void drain_secondary_chain_head();
    uint32_t get_set(uint32_t addr) const;
    uint32_t get_tag(uint32_t addr) const;
    uint32_t get_line_addr(uint32_t addr) const;
    bool any_pinned_tag() const;
    // Effective line pin: the chain pin (CacheTag::pinned) OR the write-ack
    // pin (a non-zero per-set outstanding-write count). Reads the REGISTERED
    // current_ slots only.
    bool is_pinned(uint32_t set) const;
    // True when the global enqueued-but-unacked write-through count is at the
    // max_outstanding_writes ceiling. Reads the REGISTERED current_ scalar.
    bool outstanding_writes_at_cap() const;
    // Single, fallible wrapper for all write-buffer enqueues. The write
    // buffer has one enqueue port per cycle: the first caller claims it and
    // stages the push (applied at commit()) plus the per-set / global
    // outstanding-write counter increments, returning true; a later caller
    // this cycle finds the port claimed and returns false without staging
    // anything. Callers must confirm admission (write-buffer depth AND
    // outstanding_writes_at_cap()) before invoking it, and must perform any
    // tag install / MSHR free / pin clear only on a true return.
    bool queue_write_through(uint32_t line_addr);

    uint32_t cache_size_;        // config
    uint32_t line_size_;         // config
    uint32_t num_sets_;          // config

    // REGISTERED tag array (Reg<std::vector<CacheTag>>; the whole vector is
    // one register). Readers in cycle T see tags_.current() (the value
    // latched at the start of T); fills write tags_.next_mut() and latch for
    // T+1. This retires the implicit write-first tag-array assumption.
    Reg<std::vector<CacheTag>> tags_;
    MSHRFile mshrs_;
    // REGISTERED, single-enqueue-port write-buffer FIFO (Phase 5a:
    // RegFifo<uint32_t>). Mutated only by commit(): evaluate-phase reads see
    // the committed deque. RegFifo's claim_port()/port_claimed() encodes the
    // per-cycle first-writer-wins enqueue arbitration (modeled on
    // LoadGatherBufferFile's port-claim flag). stage_push() / stage_pop()
    // queue the <=1 staged enqueue and the staged drain; commit() applies
    // pop-then-push and clears the staging slots.
    RegFifo<uint32_t> write_buffer_;
    uint32_t write_buffer_depth_;     // config
    uint32_t max_outstanding_writes_; // config
    ExternalMemoryInterface& mem_if_; // config (back-pointer)
    LoadGatherBufferFile& gather_file_;  // config (back-pointer)
    Stats& stats_;                    // config (back-pointer)
    // COMBINATIONAL same-tick scratch: reset at top of evaluate(), written
    // by handle_responses / process_load / process_store / complete_fill,
    // observed mid-tick by CoalescingUnit::evaluate. No next/current pair —
    // coalescing reads this cycle's value to make a same-cycle bail
    // decision (single-cycle backpressure path; tags_/write_buffer_/
    // pending_fill_ all source from registered state). Phase 7 (reg.h
    // migration): wrapped as Wire<bool> / Wire<CacheStallReason>. drive()
    // asserts; reset() de-asserts (default false / NONE). Not enrolled via
    // register_state (Wire is not a RegBase).
    Wire<bool> stalled_;
    Wire<CacheStallReason> stall_reason_;
    // COMBINATIONAL same-tick scratch: the set a successful (non-deferred)
    // complete_fill installed into this cycle, or -1 if none. Produced by
    // complete_fill, consumed later in the same tick by process_load /
    // process_store to reject a command racing the fill to the same set
    // (fill-conflict retry). Reset to -1 at the top of evaluate() and again
    // at commit(): its lifetime is exactly one tick, so it must not leak
    // past the tick boundary. The commit() reset is redundant in production
    // (evaluate()'s top-of-tick reset governs) but makes the field's
    // same-tick lifetime explicit for the direct-API test path. Phase 7
    // (reg.h migration): wrapped as Wire<int32_t> with a non-default value of
    // -1 (reset() restores the -1 default). Intra-class signal — no public
    // accessor; readers (process_load/process_store) read wire_.value()
    // directly.
    Wire<int32_t> fill_installed_set_{-1};
    // REGISTERED deferred-fill carrier.
    Reg<PendingCacheFill> pending_fill_;
    // Write-ack pin state, REGISTERED. outstanding_writes_[set] counts
    // enqueued-but-unacked write-throughs to that set: a queued write-through
    // pins its set until the external write ack returns. The total scalar
    // is the running sum — the state of the global max_outstanding_writes
    // cap. Enqueues increment next_; write-ack consumption decrements next_;
    // commit() flips.
    Reg<std::vector<uint32_t>> outstanding_writes_;
    Reg<uint32_t> outstanding_writes_total_;
    // REGISTERED trace event slots — written next_mut() during evaluate,
    // flipped by commit, read post-commit by TimingModel::record_cycle_trace.
    Reg<CacheMissTraceEvent> last_miss_event_;
    Reg<CacheFillTraceEvent> last_fill_event_;
    Reg<CacheSecondaryDrainTraceEvent> last_drain_event_;
    Reg<CachePinStallTraceEvent> last_pin_stall_event_;
    // Phase 7: the prior `gather_extract_port_used_` scratch flag has been
    // removed. The shared gather-extract port (spec §5.3 — one line-to-
    // gather-buffer extraction per cycle) is now arbitrated by
    // `LoadGatherBufferFile` via a single REGISTERED `next_port_claimed_` /
    // `current_port_claimed_` pair; the cache observes claim outcomes
    // through `try_write()`'s bool return. The FILL > secondary > HIT
    // priority is encoded in tick order: cache_->evaluate() runs at the top
    // of the non-panic tick (FILL first via handle_responses, secondary
    // second via drain_secondary_chain_head); coalescing_->evaluate() runs
    // later in the tick (HIT third).

    // Phase M3: REGISTERED forward command slots. Coalescing writes the
    // staged slot via set_next_load_cmd / set_next_store_cmd; commit() flips
    // next → current; evaluate() consumes the committed slot after
    // handle_responses + drain_secondary_chain_head (HIT slot in the
    // FILL > secondary > HIT priority ladder).
    //
    // Phase 4 of current_mut() elimination (Pattern 3): wrapped as
    // PulseReg<T>. PulseReg::commit() resets next_ to T{} after the flip,
    // so a cycle on which coalescing does not stage a fresh command latches
    // T{} into current_ at the next commit — the memoryless-consumer
    // contract encoded in the type. Previous shape was Reg<T> with the
    // consumer's mid-cycle current_mut().valid=false clear plus a tail-of-
    // commit set_next(T{}) reset; both are replaced by PulseReg's commit-
    // time reset.
    PulseReg<LoadCommand> load_cmd_;
    PulseReg<StoreCommand> store_cmd_;
    // Phase M3 (valid/ready): consumer-side ready signal. Reset to false
    // at top of evaluate; set true by evaluate when a cmd from the committed
    // slot was processed and accepted. Read combinationally by coalescing
    // later in the same tick. COMBINATIONAL same-tick scratch. Phase 7:
    // wrapped as Wire<bool>; default false; drive(true) on accepted cmd.
    Wire<bool> next_cmd_ready_;
};

} // namespace gpu_sim
