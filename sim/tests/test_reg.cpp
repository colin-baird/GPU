#include "catch.hpp"
#include "gpu_sim/timing/reg.h"

#include <deque>

using namespace gpu_sim;

// ─── Reg<T> ─────────────────────────────────────────────────────────────────

TEST_CASE("Reg: staged write is invisible to current() until commit", "[reg]") {
    Reg<int> r;
    REQUIRE(r.current() == 0);

    r.set_next(7);
    REQUIRE(r.current() == 0);  // committed value unchanged
    REQUIRE(r.next() == 7);     // staged value visible to the producer

    r.commit();
    REQUIRE(r.current() == 7);
}

TEST_CASE("Reg: seed re-establishes next_ = current_ and is idempotent", "[reg]") {
    Reg<int> r{3};
    REQUIRE(r.current() == 3);
    REQUIRE(r.next() == 3);

    r.set_next(99);
    r.seed();                   // discard the staged value
    REQUIRE(r.next() == 3);

    r.seed();                   // idempotent — a stalled re-seed changes nothing
    REQUIRE(r.next() == 3);
    REQUIRE(r.current() == 3);
}

TEST_CASE("Reg: a stalled cycle (seed without commit) re-runs identically",
          "[reg]") {
    Reg<int> r{5};

    // Cycle N: stage a value, but commit is gated (stalled) — skip it.
    r.seed();
    r.next_mut() += 1;
    REQUIRE(r.next() == 6);
    // commit() skipped.

    // Cycle N+1: seed re-derives from the unchanged committed state.
    r.seed();
    REQUIRE(r.next() == 5);
    r.next_mut() += 1;
    r.commit();
    REQUIRE(r.current() == 6);  // exactly one increment landed
}

TEST_CASE("Reg: next_mut allows in-place mutation of the staged value", "[reg]") {
    Reg<std::deque<int>> r;
    r.seed();
    r.next_mut().push_back(1);
    r.next_mut().push_back(2);
    REQUIRE(r.current().empty());        // committed deque untouched
    REQUIRE(r.next().size() == 2);

    r.commit();
    REQUIRE(r.current().size() == 2);
    REQUIRE(r.current().front() == 1);
}

TEST_CASE("Reg: reset clears both slots", "[reg]") {
    Reg<int> r{42};
    r.set_next(7);
    r.reset();
    REQUIRE(r.current() == 0);
    REQUIRE(r.next() == 0);
}

// ─── RegFifo<T> ──────────────────────────────────────────────────────────────

TEST_CASE("RegFifo: a staged push is applied only at commit", "[reg][regfifo]") {
    RegFifo<int> fifo;
    fifo.stage_push(11);
    REQUIRE(fifo.current_empty());       // not visible during evaluate

    fifo.commit();
    REQUIRE(fifo.current_size() == 1);
    REQUIRE(fifo.current_front() == 11);
}

TEST_CASE("RegFifo: commit applies pop before push", "[reg][regfifo]") {
    RegFifo<int> fifo;
    fifo.stage_push(1);
    fifo.commit();
    fifo.stage_push(2);
    fifo.commit();
    REQUIRE(fifo.current_size() == 2);

    // Same cycle: pop the front and push a new entry.
    fifo.stage_pop();
    fifo.stage_push(3);
    fifo.commit();

    REQUIRE(fifo.current_size() == 2);
    REQUIRE(fifo.current_front() == 2);  // 1 popped
    REQUIRE(fifo.current().back() == 3); // 3 pushed
}

TEST_CASE("RegFifo: seed is a no-op; reset clears the queue and staging",
          "[reg][regfifo]") {
    RegFifo<int> fifo;
    fifo.stage_push(1);
    fifo.commit();
    fifo.seed();                         // no-op
    REQUIRE(fifo.current_size() == 1);

    fifo.stage_push(2);                  // staged but not committed
    fifo.reset();
    REQUIRE(fifo.current_empty());
    fifo.commit();
    REQUIRE(fifo.current_empty());       // discarded staged push did not survive
}

TEST_CASE("RegFifo: single enqueue-port claim is first-come", "[reg][regfifo]") {
    RegFifo<int> fifo;
    REQUIRE_FALSE(fifo.port_claimed());
    fifo.claim_port();
    REQUIRE(fifo.port_claimed());

    fifo.commit();
    REQUIRE_FALSE(fifo.port_claimed());  // claim clears at the cycle boundary
}

// ─── Wire<T> ─────────────────────────────────────────────────────────────────

TEST_CASE("Wire: drive asserts a value, reset restores the default", "[reg][wire]") {
    Wire<bool> w;
    REQUIRE_FALSE(w.value());

    w.drive(true);
    REQUIRE(w.value());

    w.reset();
    REQUIRE_FALSE(w.value());
}

TEST_CASE("Wire: a non-default de-asserted value is honored", "[reg][wire]") {
    Wire<int> w{-1};
    REQUIRE(w.value() == -1);
    w.drive(5);
    REQUIRE(w.value() == 5);
    w.reset();
    REQUIRE(w.value() == -1);
}

// ─── RegisteredStage ─────────────────────────────────────────────────────────

namespace {
// A minimal stand-in for a timing-model stage: two Reg members and a RegFifo,
// all registered for uniform seed/commit/reset.
class FakeStage : public RegisteredStage {
public:
    FakeStage() { register_state(&a_, &b_, &fifo_); }

    void evaluate() {
        seed_all();
        a_.next_mut() += 1;
        b_.set_next(b_.current() * 2 + 1);
        fifo_.stage_push(static_cast<int>(a_.next()));
    }
    void commit_stage() { commit_all(); }

    Reg<int>& a() { return a_; }
    Reg<int>& b() { return b_; }
    RegFifo<int>& fifo() { return fifo_; }
    void reset_stage() { reset_all(); }

private:
    Reg<int> a_;
    Reg<int> b_;
    RegFifo<int> fifo_;
};
} // namespace

TEST_CASE("RegisteredStage: seed/commit/reset drive every registered primitive",
          "[reg][registered-stage]") {
    FakeStage stage;

    stage.evaluate();
    stage.commit_stage();
    REQUIRE(stage.a().current() == 1);
    REQUIRE(stage.b().current() == 1);
    REQUIRE(stage.fifo().current_size() == 1);

    stage.evaluate();
    stage.commit_stage();
    REQUIRE(stage.a().current() == 2);
    REQUIRE(stage.b().current() == 3);
    REQUIRE(stage.fifo().current_size() == 2);

    stage.reset_stage();
    REQUIRE(stage.a().current() == 0);
    REQUIRE(stage.b().current() == 0);
    REQUIRE(stage.fifo().current_empty());
}

TEST_CASE("RegisteredStage: a skipped commit_all freezes all registers",
          "[reg][registered-stage]") {
    FakeStage stage;
    stage.evaluate();
    stage.commit_stage();           // a=1, b=1

    stage.evaluate();               // stages a=2, b=3 ...
    // ... commit gated (stalled): skip commit_all().
    REQUIRE(stage.a().current() == 1);
    REQUIRE(stage.b().current() == 1);

    stage.evaluate();               // re-seed re-derives from committed state
    stage.commit_stage();
    REQUIRE(stage.a().current() == 2);  // exactly one increment landed
    REQUIRE(stage.b().current() == 3);
}
