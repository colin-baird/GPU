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

// ─── PulseReg<T> ─────────────────────────────────────────────────────────────

TEST_CASE("PulseReg: staged write is invisible to current() until commit",
          "[reg][pulsereg]") {
    PulseReg<int> p;
    REQUIRE(p.current() == 0);

    p.set_next(7);
    REQUIRE(p.current() == 0);   // committed slot unchanged
    REQUIRE(p.next() == 7);      // staged value visible to the producer

    p.commit();
    REQUIRE(p.current() == 7);
}

TEST_CASE("PulseReg: seed defaults next_ to T{} (key difference from Reg)",
          "[reg][pulsereg]") {
    PulseReg<int> p;
    p.set_next(99);
    REQUIRE(p.next() == 99);
    p.commit();
    REQUIRE(p.current() == 99);

    // Seed: next_ goes to T{}, NOT current_ (a Reg would copy current_ here).
    p.seed();
    REQUIRE(p.next() == 0);
    REQUIRE(p.current() == 99);  // committed slot unchanged by seed
}

TEST_CASE("PulseReg: no-driver cycle defaults current() to T{} at next commit",
          "[reg][pulsereg]") {
    PulseReg<int> p;

    // Cycle N: producer drives a value.
    p.seed();
    p.set_next(42);
    p.commit();
    REQUIRE(p.current() == 42);

    // Cycle N+1: producer does NOT drive — seed defaults next_ to 0, commit
    // latches the default. This is the pulse semantics: a value present only
    // for the cycle the producer actively staged it.
    p.seed();
    p.commit();
    REQUIRE(p.current() == 0);
}

TEST_CASE("PulseReg: a stalled cycle (seed without commit) re-defaults",
          "[reg][pulsereg]") {
    PulseReg<int> p;

    // Cycle N: producer stages, but commit is gated.
    p.seed();
    p.set_next(5);
    REQUIRE(p.next() == 5);
    // commit() skipped.

    // Cycle N+1: seed re-defaults next_ to 0. The producer must re-assert if
    // it still wants the value to latch.
    p.seed();
    REQUIRE(p.next() == 0);
    p.set_next(5);
    p.commit();
    REQUIRE(p.current() == 5);
}

TEST_CASE("PulseReg: next_mut allows in-place mutation of the staged value",
          "[reg][pulsereg]") {
    struct Cmd { bool valid = false; int payload = 0; };
    PulseReg<Cmd> p;
    p.seed();

    p.next_mut().valid = true;
    p.next_mut().payload = 11;
    REQUIRE_FALSE(p.current().valid);   // committed slot untouched
    REQUIRE(p.next().valid);
    REQUIRE(p.next().payload == 11);

    p.commit();
    REQUIRE(p.current().valid);
    REQUIRE(p.current().payload == 11);
}

TEST_CASE("PulseReg: reset clears both slots", "[reg][pulsereg]") {
    PulseReg<int> p{42};
    p.set_next(7);
    p.reset();
    REQUIRE(p.current() == 0);
    REQUIRE(p.next() == 0);
}

TEST_CASE("PulseReg: initialize seeds both slots to the same value",
          "[reg][pulsereg]") {
    PulseReg<int> p;
    p.initialize(99);
    REQUIRE(p.current() == 99);
    REQUIRE(p.next() == 99);
}

TEST_CASE("PulseReg: producer/consumer pulse pattern across multiple cycles",
          "[reg][pulsereg]") {
    // The canonical usage: producer drives only on cycles it has a command;
    // consumer reads current(), processes, and never modifies the slot. The
    // slot's default-to-T{} between drives is what eliminates the need for
    // the consumer to clear committed state mid-cycle.
    struct Cmd { bool valid = false; int payload = 0; };
    PulseReg<Cmd> p;

    // Cycle 0: no producer drive.
    p.seed();
    REQUIRE_FALSE(p.current().valid);   // consumer reads — nothing to do
    p.commit();
    REQUIRE_FALSE(p.current().valid);

    // Cycle 1: producer drives a command.
    p.seed();
    p.set_next({/*valid=*/true, /*payload=*/77});
    p.commit();

    // Cycle 2: consumer sees the command in current(); producer is silent.
    p.seed();
    REQUIRE(p.current().valid);
    REQUIRE(p.current().payload == 77);
    p.commit();

    // Cycle 3: the command has lapsed — no carry-forward.
    p.seed();
    REQUIRE_FALSE(p.current().valid);
    p.commit();
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

TEST_CASE("RegFifo: stage_pop counts and applies multiple pops at commit",
          "[reg][regfifo]") {
    RegFifo<int> fifo;
    // Seed three entries.
    fifo.stage_push(10);
    fifo.commit();
    fifo.stage_push(20);
    fifo.commit();
    fifo.stage_push(30);
    fifo.commit();
    REQUIRE(fifo.current_size() == 3);

    // Stage three pops in one cycle (Phase 5a — the multi-pop counter).
    fifo.stage_pop();
    REQUIRE(fifo.pops_staged() == 1);
    fifo.stage_pop();
    fifo.stage_pop();
    REQUIRE(fifo.pops_staged() == 3);
    // Reads during evaluate still see start-of-cycle state.
    REQUIRE(fifo.current_size() == 3);
    REQUIRE(fifo.current_front() == 10);

    fifo.commit();
    REQUIRE(fifo.current_empty());
    // pops_ counter clears at commit.
    REQUIRE(fifo.pops_staged() == 0);
}

TEST_CASE("RegFifo: pops_ is clamped to queue size at commit",
          "[reg][regfifo]") {
    RegFifo<int> fifo;
    fifo.stage_push(42);
    fifo.commit();
    REQUIRE(fifo.current_size() == 1);

    // Over-stage pops (e.g. a DRAM-clock stage that ran ClockTick more times
    // than the queue had entries this cycle).
    fifo.stage_pop();
    fifo.stage_pop();
    fifo.stage_pop();
    fifo.commit();
    REQUIRE(fifo.current_empty());           // not a crash, not undefined
    REQUIRE(fifo.pops_staged() == 0);
}

TEST_CASE("RegFifo: reset clears the multi-pop counter", "[reg][regfifo]") {
    RegFifo<int> fifo;
    fifo.stage_push(1);
    fifo.commit();
    fifo.stage_pop();
    fifo.stage_pop();
    REQUIRE(fifo.pops_staged() == 2);
    fifo.reset();
    REQUIRE(fifo.pops_staged() == 0);
    REQUIRE(fifo.current_empty());
}

TEST_CASE("RegFifo: pops_staged enables peek-ahead during multi-pop staging",
          "[reg][regfifo]") {
    RegFifo<int> fifo;
    fifo.stage_push(7);
    fifo.commit();
    fifo.stage_push(8);
    fifo.commit();
    fifo.stage_push(9);
    fifo.commit();

    // A DRAM-clock-style consumer stages pops one at a time and peeks at the
    // next-to-pop entry via queue_[pops_staged()] before deciding whether to
    // stage another. current() returns the committed deque so the consumer
    // can index past the already-staged head pops.
    REQUIRE(fifo.current()[fifo.pops_staged()] == 7);
    fifo.stage_pop();
    REQUIRE(fifo.current()[fifo.pops_staged()] == 8);
    fifo.stage_pop();
    REQUIRE(fifo.current()[fifo.pops_staged()] == 9);
    fifo.stage_pop();
    REQUIRE(fifo.pops_staged() == 3);

    fifo.commit();
    REQUIRE(fifo.current_empty());
}

TEST_CASE("RegFifo: stage_push accumulates and applies multiple pushes at commit",
          "[reg][regfifo]") {
    RegFifo<int> fifo;
    // Phase 5 (sparkling-dazzling-starfish.md): the multi-push extension
    // lets a producer that may push more than one value per cycle (the
    // DRAMSim3 fabric-clock stage emitting chunks_per_line PendingChunks
    // per submit_*, the DRAM-clock stage emitting one ack per ClockTick)
    // stage them all in one cycle. pushes_staged() reports the count.
    fifo.stage_push(1);
    REQUIRE(fifo.pushes_staged() == 1u);
    fifo.stage_push(2);
    fifo.stage_push(3);
    REQUIRE(fifo.pushes_staged() == 3u);
    // Reads during evaluate still see start-of-cycle state (the staging
    // deque is separate from queue_).
    REQUIRE(fifo.current_empty());

    fifo.commit();
    REQUIRE(fifo.current_size() == 3);
    // FIFO order — pushes applied in stage order.
    REQUIRE(fifo.current()[0] == 1);
    REQUIRE(fifo.current()[1] == 2);
    REQUIRE(fifo.current()[2] == 3);
    // pushes_staged clears at commit.
    REQUIRE(fifo.pushes_staged() == 0u);
}

TEST_CASE("RegFifo: same-cycle multi-pop + multi-push commits correctly",
          "[reg][regfifo]") {
    RegFifo<int> fifo;
    // Seed three entries.
    fifo.stage_push(10);
    fifo.stage_push(20);
    fifo.stage_push(30);
    fifo.commit();
    REQUIRE(fifo.current_size() == 3);

    // Same cycle: pop the front two, push two new entries.
    fifo.stage_pop();
    fifo.stage_pop();
    fifo.stage_push(40);
    fifo.stage_push(50);
    fifo.commit();

    REQUIRE(fifo.current_size() == 3);
    // Pops apply first (FIFO order), then pushes.
    REQUIRE(fifo.current()[0] == 30);
    REQUIRE(fifo.current()[1] == 40);
    REQUIRE(fifo.current()[2] == 50);
}

TEST_CASE("RegFifo: reset discards staged pushes", "[reg][regfifo]") {
    RegFifo<int> fifo;
    fifo.stage_push(1);
    fifo.stage_push(2);
    REQUIRE(fifo.pushes_staged() == 2u);
    fifo.reset();
    REQUIRE(fifo.pushes_staged() == 0u);
    fifo.commit();
    REQUIRE(fifo.current_empty());
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

TEST_CASE("RegisteredStage: enrolls PulseReg alongside Reg / RegFifo",
          "[reg][registered-stage][pulsereg]") {
    struct Cmd { bool valid = false; int payload = 0; };

    class MixedStage : public RegisteredStage {
    public:
        MixedStage() { register_state(&counter_, &cmd_); }

        // Drive the pulse only on even-numbered evaluate calls.
        void evaluate(int tick) {
            seed_all();
            counter_.next_mut() += 1;
            if (tick % 2 == 0) {
                cmd_.set_next({/*valid=*/true, /*payload=*/tick});
            }
            // Odd ticks: producer silent — PulseReg defaults next_ to T{}
            // via seed_all() above, so commit will latch invalid.
        }
        void commit_stage() { commit_all(); }
        void reset_stage() { reset_all(); }

        Reg<int>& counter() { return counter_; }
        PulseReg<Cmd>& cmd() { return cmd_; }

    private:
        Reg<int> counter_;
        PulseReg<Cmd> cmd_;
    };

    MixedStage stage;

    stage.evaluate(0);  // drives cmd
    stage.commit_stage();
    REQUIRE(stage.counter().current() == 1);
    REQUIRE(stage.cmd().current().valid);
    REQUIRE(stage.cmd().current().payload == 0);

    stage.evaluate(1);  // does NOT drive cmd
    stage.commit_stage();
    REQUIRE(stage.counter().current() == 2);
    REQUIRE_FALSE(stage.cmd().current().valid);

    stage.evaluate(2);  // drives cmd again
    stage.commit_stage();
    REQUIRE(stage.counter().current() == 3);
    REQUIRE(stage.cmd().current().valid);
    REQUIRE(stage.cmd().current().payload == 2);

    stage.reset_stage();
    REQUIRE(stage.counter().current() == 0);
    REQUIRE_FALSE(stage.cmd().current().valid);
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
