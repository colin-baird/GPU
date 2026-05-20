#pragma once

#include <cstddef>
#include <deque>
#include <vector>

namespace gpu_sim {

// ─── State primitives for the cycle-accurate timing model ───────────────────
//
// The timing model mimics synchronous RTL: each stage's evaluate() computes a
// staged ("next") value from committed ("current") state, and commit() latches
// next -> current at the cycle boundary. These primitives make that discipline
// the only way to express state — a same-cycle-visible mutation (a plain
// member written and read within one cycle, a hazard in real hardware) cannot
// be expressed by accident.
//
// Every state-holding member of a timing-model class is exactly one of:
//   - Reg<T>      a clock-edge-latched register (holds across cycles)
//   - PulseReg<T> a one-cycle pulse / command slot (defaults to T{} each
//                 cycle unless the producer explicitly stages a value)
//   - RegFifo<T>  a commit-disciplined FIFO
//   - Wire<T>     a combinational backward signal (no committed twin)
// plus plain members that are either config (const after construction) or
// sim-instrumentation (observational counters; annotated // sim-instrumentation).
//
// See resources/timing_discipline.md and resources/cpp_coding_standard.md.

// Non-template base so a stage can hold heterogeneous registers in one list
// (RegisteredStage) and drive seed/commit/reset uniformly.
class RegBase {
public:
    virtual ~RegBase() = default;
    virtual void seed() = 0;
    virtual void commit() = 0;
    virtual void reset() = 0;
};

// A clock-edge-latched register. evaluate() reads current() (committed) and
// stages the new value via set_next() / next_mut(); commit() latches it.
// seed() re-establishes next_ = current_ at the top of every tick so a stalled
// re-evaluated cycle (commit skipped) re-runs identically.
template <typename T>
class Reg : public RegBase {
public:
    Reg() = default;
    explicit Reg(const T& init) : current_(init), next_(init) {}

    // Committed read — the normal cross-cycle / cross-stage read.
    const T& current() const { return current_; }

    // Committed read-write. Narrow escape hatch for the five patterns that
    // legitimately need a mutation of the committed slot outside the normal
    // staged-write-then-commit flow. Use only at a documented site:
    //   (1) Redirect-flush: a backward control signal (a branch mispredict
    //       resolved this same cycle) forces an upstream stage to invalidate
    //       committed state mid-evaluate() — the flush is the redirect's
    //       same-cycle effect (e.g. FetchStage clearing current_output_).
    //   (2) Post-commit consumed-mark: a stage's commit() applies its
    //       pop-then-push, then marks the now-consumed slot invalid in the
    //       committed deque (e.g. DecodeStage clearing pending_.valid after
    //       the buffer push). Equivalent to a mid-cycle write because no
    //       reader has run since the commit_all() flip.
    //   (3) Memoryless-consumer mid-evaluate invalidation: a stage whose
    //       Reg is a single-cycle command slot (the producer must re-stage
    //       every cycle) clears the committed slot mid-evaluate() after
    //       consuming it (e.g. L1Cache load_cmd_/store_cmd_; the memory
    //       interfaces' read_request_/write_request_; LoadGatherBufferFile
    //       claim_request_). Paired with a tail-of-commit set_next(T{}) to
    //       re-clear the staged slot and a deliberate seed-phase opt-out
    //       (this Reg is NOT seeded by tick()'s seed phase).
    //   (4) Deferred-claim dual-write: a stage's evaluate() applies a
    //       committed-state claim into BOTH the committed and staged slots
    //       so a same-cycle reader of the committed view (e.g. coalescing's
    //       current_busy()) observes the claim immediately, AND the claim
    //       survives the commit-phase flip (e.g. LoadGatherBufferFile).
    //   (5) Test-hook dual-write: a test pre-arms state through a public
    //       hook (test_set_unit_busy / test_reserve_writeback_slot) that
    //       writes both current_ and next_ so the subsequent evaluate()
    //       observes the armed value regardless of whether the test calls
    //       seed_next() in between.
    // Not a normal staged write — ordinary updates stage via set_next() /
    // next_mut() and latch at commit().
    T& current_mut() { return current_; }

    // Staged read. INTRA-STAGE self-reads only: a producer reading back a
    // value it staged earlier in this same evaluate(). A cross-module next()
    // read is a combinational-forward hazard — forbidden, lint-enforced.
    const T& next() const { return next_; }

    // Staged read-write, for in-place mutation of the staged value (container
    // push, struct-field assignment, counter decrement) without a whole-value
    // copy.
    T& next_mut() { return next_; }

    // Staged whole-value write.
    void set_next(const T& value) { next_ = value; }

    // next_ = current_  — top-of-tick reseed; idempotent.
    void seed() override { next_ = current_; }

    // current_ = next_  — cycle-boundary latch.
    void commit() override { current_ = next_; }

    // Reset BOTH slots so a post-reset evaluate() before the first seed()
    // never observes a stale staged value.
    void reset() override { current_ = T{}; next_ = T{}; }

    // Initialize BOTH slots to the same value — for constructor / reset paths
    // that need to size a Reg<std::vector<...>> or seed a non-default value
    // before any evaluate runs. Equivalent to `set_next(value); commit();`
    // but doesn't require a transient half-state.
    void initialize(const T& value) { current_ = value; next_ = value; }

private:
    T current_{};
    T next_{};
};

// A one-cycle pulse / command-slot register. Same shape as Reg<T> except seed()
// resets next_ to T{} rather than copying current_. The producer must
// explicitly drive next_ every cycle it wants a value to latch; if it does
// nothing, the slot defaults to T{} at the next commit. The consumer reads
// current_ (the previously-latched pulse) and does NOT mutate it — the slot
// either carries forward to next cycle as T{} (no driver this cycle) or as
// the producer's new value (driver staged this cycle).
//
// Use for: memoryless-consumer command slots where the producer's silence
// must mean "no command next cycle" — cache load_cmd_/store_cmd_, memory
// backend read_request_/write_request_, gather-buffer claim_request_. These
// previously used Reg<T> with an explicit current_mut().valid=false clear by
// the consumer and a tail-of-commit set_next(T{}) clear by the same consumer
// to keep the slot from re-latching; PulseReg encodes the same default-to-T{}
// semantics in the type and removes the need for either workaround.
//
// In hardware terms: a register whose D input has a multiplexer with a "no
// driver" default of T{} — the seed-to-T{} models the multiplexer; the
// producer's set_next(v) is asserting the override.
template <typename T>
class PulseReg : public RegBase {
public:
    PulseReg() = default;
    explicit PulseReg(const T& init) : current_(init) {}

    // Committed read — the previously-latched pulse (T{} if no driver last
    // cycle). The normal read.
    const T& current() const { return current_; }

    // Staged read — INTRA-STAGE self-reads only, same discipline as Reg<T>.
    // The producer reads back what it has staged earlier in this evaluate.
    const T& next() const { return next_; }

    // Staged read-write, for in-place mutation of the staged value without a
    // whole-value copy.
    T& next_mut() { return next_; }

    // Staged whole-value write — the producer's "drive D" action.
    void set_next(const T& value) { next_ = value; }

    // next_ = T{}  — top-of-tick default-to-invalid. The producer must
    // re-assert to keep the slot non-default at the next commit. Idempotent.
    void seed() override { next_ = T{}; }

    // current_ = next_  — cycle-boundary latch. Identical to Reg<T>.
    void commit() override { current_ = next_; }

    // Reset BOTH slots to T{}.
    void reset() override { current_ = T{}; next_ = T{}; }

    // Initialize BOTH slots to the same value — for constructor / reset paths
    // that need to seed a non-default value before any evaluate runs.
    void initialize(const T& value) { current_ = value; next_ = value; }

private:
    T current_{};
    T next_{};
};

// A commit-disciplined FIFO. evaluate() stages a push and/or a pop intent;
// commit() applies pop-then-push. The committed deque is the only state, so
// seed() is a no-op. Reads during evaluate() see start-of-cycle state.
template <typename T>
class RegFifo : public RegBase {
public:
    // Committed reads — stable across the whole evaluate phase.
    const std::deque<T>& current() const { return queue_; }
    bool current_empty() const { return queue_.empty(); }
    std::size_t current_size() const { return queue_.size(); }
    const T& current_front() const { return queue_.front(); }

    // Stage a push / pop to be applied at commit().
    void stage_push(const T& value) { push_ = value; has_push_ = true; }
    void stage_pop() { pop_ = true; }

    // Single-enqueue-port claim — first claimer this cycle wins. The producer
    // checks port_claimed() before staging a push.
    bool port_claimed() const { return port_claimed_; }
    void claim_port() { port_claimed_ = true; }

    void seed() override {}  // the committed deque is the only state

    void commit() override {
        if (pop_ && !queue_.empty()) queue_.pop_front();
        if (has_push_) queue_.push_back(push_);
        has_push_ = false;
        pop_ = false;
        port_claimed_ = false;
    }

    void reset() override {
        queue_.clear();
        has_push_ = false;
        pop_ = false;
        port_claimed_ = false;
    }

private:
    std::deque<T> queue_;
    T push_{};
    bool has_push_ = false;
    bool pop_ = false;
    bool port_claimed_ = false;
};

// A combinational backward signal: asserted by a downstream stage during its
// evaluate(), read backward the same cycle by an upstream stage. It has no
// committed twin and no commit() — it is NOT a register. reset() is called at
// the top of the producing stage's evaluate(), re-establishing the default
// (de-asserted) value. The back-to-front evaluate sweep guarantees the
// producer drives the wire before any upstream consumer reads it.
template <typename T>
class Wire {
public:
    Wire() = default;
    explicit Wire(const T& default_value)
        : default_(default_value), value_(default_value) {}

    // Read the signal asserted this cycle (or the default if un-driven).
    const T& value() const { return value_; }

    // Assert the signal — called in the producer's evaluate().
    void drive(const T& value) { value_ = value; }

    // De-assert — called at the top of the producer's evaluate() and by the
    // owning stage's reset().
    void reset() { value_ = default_; }

private:
    T default_{};
    T value_{};
};

// Mixin for any timing-model class that owns Reg / RegFifo state. The class
// calls register_state(&r1, &r2, ...) once in its constructor body — the body
// runs after every member initializer, so member declaration order is
// irrelevant and there is no construction-order hazard. seed_all() /
// commit_all() / reset_all() then drive every registered primitive, so
// "forgot to seed/commit a newly added field" cannot happen.
//
// RegisteredStage-derived classes are non-copyable and non-movable: the
// registered pointers refer into this object. They live as direct members or
// behind a unique_ptr and are never copied.
class RegisteredStage {
public:
    RegisteredStage() = default;
    RegisteredStage(const RegisteredStage&) = delete;
    RegisteredStage& operator=(const RegisteredStage&) = delete;
    RegisteredStage(RegisteredStage&&) = delete;
    RegisteredStage& operator=(RegisteredStage&&) = delete;

    void seed_all()   { for (RegBase* r : regs_) r->seed(); }
    void commit_all() { for (RegBase* r : regs_) r->commit(); }
    void reset_all()  { for (RegBase* r : regs_) r->reset(); }

protected:
    ~RegisteredStage() = default;

    template <typename... Regs>
    void register_state(Regs*... regs) {
        (regs_.push_back(regs), ...);
    }

private:
    std::vector<RegBase*> regs_;
};

} // namespace gpu_sim
