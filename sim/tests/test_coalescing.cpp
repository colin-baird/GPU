#include "catch.hpp"
#include "gpu_sim/timing/coalescing_unit.h"
#include "gpu_sim/timing/ldst_unit.h"
#include "gpu_sim/timing/cache.h"
#include "gpu_sim/timing/load_gather_buffer.h"
#include "gpu_sim/timing/memory_interface.h"
#include "gpu_sim/stats.h"

using namespace gpu_sim;

namespace {

constexpr uint32_t LINE_SIZE = 128;
constexpr uint32_t CACHE_SIZE = 4096;
constexpr uint32_t NUM_MSHRS = 32; // large — serialized loads need 32 MSHRs
constexpr uint32_t WB_DEPTH = 32;
constexpr uint32_t MAX_OUTSTANDING_WRITES = 64;
constexpr uint32_t NUM_WARPS = 4;
constexpr uint32_t MEM_LATENCY = 4;

// Build a DispatchInput for a load instruction with given per-lane addresses.
DispatchInput make_load_dispatch(uint32_t warp_id, uint8_t dest_reg,
                                 const std::array<uint32_t, WARP_SIZE>& addrs) {
    DispatchInput input;
    input.warp_id = warp_id;
    input.pc = 0;
    input.decoded.type = InstructionType::LOAD;
    input.decoded.target_unit = ExecUnit::LDST;
    input.decoded.has_rd = true;
    input.decoded.rd = dest_reg;
    input.decoded.mem_op = MemOp::LW;
    input.trace.warp_id = warp_id;
    input.trace.is_load = true;
    input.trace.is_store = false;
    input.trace.mem_addresses = addrs;
    input.trace.decoded.rd = dest_reg;
    input.trace.decoded.has_rd = true;
    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        input.trace.results[i] = 100 + i;
        input.trace.mem_size[i] = 4;
    }
    return input;
}

// Bring the ldst unit to the state where its FIFO has a pending entry.
void settle_ldst(LdStUnit& ldst) {
    for (int i = 0; i < 16 && ldst.current_fifo_empty(); ++i) {
        ldst.evaluate();
        ldst.commit();
    }
}

struct CoalFixture {
    Stats stats;
    FixedLatencyMemory mem_if{MEM_LATENCY, stats};
    LoadGatherBufferFile gather_file{NUM_WARPS, stats};
    L1Cache cache{CACHE_SIZE, LINE_SIZE, NUM_MSHRS, WB_DEPTH, MAX_OUTSTANDING_WRITES,
                  mem_if, gather_file, stats};
    LdStUnit ldst{8, 4, stats};
    CoalescingUnit coal{ldst, cache, gather_file, LINE_SIZE, stats};

    // Phase M3: drive one cycle of the memory pipeline. Order mirrors
    // TimingModel::tick() — gather_file.evaluate (apply REGISTERED claim)
    // → cache.evaluate (FILL + secondary + REGISTERED cmd processing) →
    // coalescing.evaluate (stage cmd) → mem_if.evaluate → drain_write_buffer
    // → commits.
    //
    // Phase 10B.0.5: this fixture drives only the *downstream* memory
    // pipeline; settle_ldst() has already pushed the accepted op into ldst's
    // addr-gen FIFO and tick() never calls ldst.evaluate(). ldst.commit() is
    // therefore not invoked here: under the explicit double-buffering
    // convention the LdSt next_push_ staging slot is cleared at the top of
    // evaluate() (not in commit()), so calling commit() without a preceding
    // evaluate() would re-apply settle_ldst()'s last staged push every tick.
    // Under the prior convention ldst.commit() here was a no-op (the last
    // settle_ldst commit had self-cleared next_push_), so dropping it is
    // byte-identical.
    void tick() {
        gather_file.evaluate();
        cache.evaluate();
        coal.evaluate();
        mem_if.evaluate();
        cache.drain_write_buffer();
        coal.commit();
        cache.commit();
        mem_if.commit();
        gather_file.commit();
    }
};

} // namespace

TEST_CASE("Coalescing: coalesced load issues one cache request and fills all 32 slots",
          "[coalescing]") {
    CoalFixture f;

    std::array<uint32_t, WARP_SIZE> addrs;
    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        addrs[i] = 256 + (i * 4); // all in one 128-byte line
    }
    auto input = make_load_dispatch(0, 5, addrs);
    f.ldst.accept(input, 1);
    settle_ldst(f.ldst);
    REQUIRE_FALSE(f.ldst.current_fifo_empty());

    // Tick 1: coalescing pops the FIFO, claims the gather buffer, and
    // stages the REGISTERED load cmd. coalesced_requests increments here.
    f.tick();
    REQUIRE(f.stats.coalesced_requests == 1);
    REQUIRE(f.stats.serialized_requests == 0);

    // Tick 2: gather_file applies the claim (buf.busy=true); cache processes
    // the REGISTERED cmd, takes a miss, allocates an MSHR. load_misses
    // increments here.
    f.tick();
    REQUIRE(f.stats.load_misses == 1);
    REQUIRE(f.cache.active_mshr_count() == 1);
    REQUIRE(f.gather_file.current_busy(0));

    // Pump until the fill arrives and lands in the gather buffer.
    for (uint32_t i = 0; i < MEM_LATENCY + 4 && !f.gather_file.current_has_result(); ++i) {
        f.tick();
    }
    REQUIRE(f.gather_file.current_has_result());
    REQUIRE(f.gather_file.buffer(0).filled_count == WARP_SIZE);

    WritebackEntry wb = f.gather_file.consume_result();
    REQUIRE(wb.dest_reg == 5);
    REQUIRE(wb.values[0] == 100);
    REQUIRE(wb.values[31] == 131);
}

TEST_CASE("Coalescing: scattered addresses serialize to 32 cache requests",
          "[coalescing]") {
    CoalFixture f;

    std::array<uint32_t, WARP_SIZE> addrs;
    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        addrs[i] = i * LINE_SIZE * 100; // each lane in a distinct line
    }
    auto input = make_load_dispatch(0, 6, addrs);
    f.ldst.accept(input, 1);
    settle_ldst(f.ldst);
    REQUIRE_FALSE(f.ldst.current_fifo_empty());

    // Tick 1: coalescing pops FIFO, claims buffer, stages lane-0 cmd.
    // serialized_requests increments here.
    f.tick();
    REQUIRE(f.stats.serialized_requests == 1);
    REQUIRE(f.stats.coalesced_requests == 0);

    // Drive ticks until coalescing finishes serializing all 32 lanes and
    // all 32 cache cmds have been processed (load_misses == WARP_SIZE).
    // Phase M3: each cmd takes one cycle to traverse coalescing → cache
    // (REGISTERED), so 32 cmds take ~33 ticks plus settling.
    for (uint32_t step = 0;
         step < 256 && (f.stats.load_misses < WARP_SIZE || !f.coal.is_idle());
         ++step) {
        f.tick();
    }
    REQUIRE(f.stats.load_misses == WARP_SIZE);
    REQUIRE(f.gather_file.current_busy(0));

    // Continue ticking to let fills come back and populate the gather buffer.
    for (uint32_t i = 0; i < 8 * MEM_LATENCY + WARP_SIZE; ++i) {
        f.tick();
        if (f.gather_file.buffer(0).filled_count == WARP_SIZE) break;
    }
    REQUIRE(f.gather_file.buffer(0).filled_count == WARP_SIZE);
    REQUIRE(f.gather_file.current_has_result());

    WritebackEntry wb = f.gather_file.consume_result();
    REQUIRE(wb.dest_reg == 6);
}

TEST_CASE("Coalescing: boundary case — one lane in a different line serializes",
          "[coalescing]") {
    CoalFixture f;

    std::array<uint32_t, WARP_SIZE> addrs;
    for (uint32_t i = 0; i < WARP_SIZE - 1; ++i) {
        addrs[i] = i * 4;
    }
    addrs[WARP_SIZE - 1] = LINE_SIZE; // a different cache line

    auto input = make_load_dispatch(0, 7, addrs);
    f.ldst.accept(input, 1);
    settle_ldst(f.ldst);

    f.tick();
    REQUIRE(f.stats.serialized_requests == 1);
    REQUIRE(f.stats.coalesced_requests == 0);
}

TEST_CASE("Coalescing: store serialization walks 32 lanes without gather buffer use",
          "[coalescing]") {
    CoalFixture f;

    std::array<uint32_t, WARP_SIZE> addrs;
    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        addrs[i] = i * LINE_SIZE * 100; // distinct lines → serialized
    }
    DispatchInput input;
    input.warp_id = 0;
    input.pc = 0;
    input.decoded.type = InstructionType::STORE;
    input.decoded.target_unit = ExecUnit::LDST;
    input.decoded.mem_op = MemOp::SW;
    input.trace.warp_id = 0;
    input.trace.is_load = false;
    input.trace.is_store = true;
    input.trace.mem_addresses = addrs;
    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        input.trace.mem_size[i] = 4;
        input.trace.store_data[i] = i;
    }

    f.ldst.accept(input, 1);
    settle_ldst(f.ldst);

    // Drive the full memory pipeline (M3 tick helper) until coalescing
    // and cache have drained. The loop runs while any work remains in the
    // chain — FIFO non-empty, coal busy, cache busy.
    for (uint32_t step = 0;
         step < 512 &&
         (!f.coal.is_idle() || !f.ldst.current_fifo_empty() || !f.cache.is_idle());
         ++step) {
        f.tick();
    }

    // Stores never claim the gather buffer.
    REQUIRE_FALSE(f.gather_file.current_busy(0));
    REQUIRE_FALSE(f.gather_file.current_has_result());
    REQUIRE(f.stats.store_misses == WARP_SIZE);
    REQUIRE(f.stats.coalesced_requests == 0);
    REQUIRE(f.stats.serialized_requests == 1);
}
