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

    // current_mut() does not exist on this type and will not be added.
    // A register's Q output does not change between clock edges, so a
    // mid-cycle write to the committed slot has no synthesis analog. The
    // five patterns that previously needed an escape hatch are each
    // replaced by a hardware-faithful encoding:
    //   (1) Redirect-flush -> combinational gate at the consumer (mask its
    //       read of Q against the same redirect Wire).
    //   (2) Post-commit consumed-mark -> stage both the push and the
    //       pending-clear at evaluate; the per-warp instr_buffer.commit()
    //       latches both together.
    //   (3) Memoryless-consumer slots -> PulseReg<T>, whose commit()
    //       resets next_ to T{} after the flip. A cycle on which the
    //       producer is silent latches T{} into current_.
    //   (4) Deferred-claim dual-write -> Wire<bitset>; current_busy() ORs
    //       the committed flag with the same-cycle wire.
    //   (5) Test-hook dual-write -> tests call seed_next() before arming
    //       the staged slot.
    // History: this hatch was removed in the current_mut()-elimination
    // refactor (project-plans/goofy-humming-dream.md).

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

// A one-cycle pulse / command-slot register. Same shape as Reg<T> except:
//   - seed()   resets next_ to T{} (rather than copying current_).
//   - commit() resets next_ to T{} AFTER the current_ = next_ flip.
// The two resets together encode the pulse semantics: the producer must
// explicitly drive next_ every cycle it wants a value to latch; if it does
// nothing, the slot defaults to T{} at the next commit. The consumer reads
// current_ (the previously-latched pulse) and does NOT mutate it.
//
// The commit-time reset is what makes the type test-friendly: a test that
// does set_next() → commit() → evaluate() without calling seed() in between
// still observes the slot defaulting to T{} on the cycle after the commit,
// matching what TimingModel::tick() achieves in production via its seed
// phase. The seed() method remains for idempotent re-seeding on stalled
// re-evaluation (its effect is a no-op when commit() has just run, since
// commit already cleared next_).
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
// driver" default of T{} — the seed and commit resets model the multiplexer;
// the producer's set_next(v) is asserting the override.
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

    // current_ = next_, then next_ = T{}  — cycle-boundary latch followed
    // by the default-to-invalid reset of the staged slot. The reset means
    // a subsequent commit() with no intervening set_next() latches T{},
    // encoding the pulse semantics directly in commit() and obviating any
    // need for a separately-invoked seed step.
    void commit() override { current_ = next_; next_ = T{}; }

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
