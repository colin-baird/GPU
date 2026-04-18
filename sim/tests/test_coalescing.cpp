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
    for (int i = 0; i < 16 && ldst.fifo_empty(); ++i) {
        ldst.evaluate();
        ldst.commit();
    }
}

struct CoalFixture {
    Stats stats;
    ExternalMemoryInterface mem_if{MEM_LATENCY, stats};
    LoadGatherBufferFile gather_file{NUM_WARPS, stats};
    L1Cache cache{CACHE_SIZE, LINE_SIZE, NUM_MSHRS, WB_DEPTH, mem_if, gather_file, stats};
    LdStUnit ldst{8, 4, stats};
    CoalescingUnit coal{ldst, cache, gather_file, LINE_SIZE, stats};
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
    REQUIRE_FALSE(f.ldst.fifo_empty());

    // One coalescing step should pop the FIFO, claim the gather buffer, and
    // issue a single cache load request for the whole warp.
    f.coal.evaluate();
    f.coal.commit();
    REQUIRE(f.stats.coalesced_requests == 1);
    REQUIRE(f.stats.serialized_requests == 0);
    REQUIRE(f.stats.load_misses == 1);
    REQUIRE(f.cache.active_mshr_count() == 1);
    REQUIRE(f.gather_file.is_busy(0));
    f.gather_file.commit();

    // Wait for the fill to complete and land in the gather buffer.
    for (uint32_t i = 0; i < MEM_LATENCY; ++i) {
        f.mem_if.evaluate();
    }
    f.cache.evaluate();
    f.cache.handle_responses();
    REQUIRE(f.gather_file.has_result());
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
    REQUIRE_FALSE(f.ldst.fifo_empty());

    // First step: pop FIFO, claim buffer, issue lane-0 request.
    f.coal.evaluate();
    REQUIRE(f.stats.serialized_requests == 1);
    REQUIRE(f.stats.coalesced_requests == 0);
    REQUIRE(f.gather_file.is_busy(0));
    f.gather_file.commit();

    // Drive the remaining 31 serialized requests through. Each cycle the
    // coalescing unit issues one more lane; memory ticks in lockstep.
    for (uint32_t step = 0; step < 64 && !f.coal.is_idle(); ++step) {
        f.coal.evaluate();
        f.cache.handle_responses();
        f.mem_if.evaluate();
        f.gather_file.commit();
    }

    // Serialization must emit exactly one cache transaction per lane.
    REQUIRE(f.stats.load_misses == WARP_SIZE);

    // Gather buffer fills incrementally; drain completed fills. Writeback
    // must not appear until all 32 slots are valid. Note: cache.evaluate()
    // internally runs handle_responses() — calling both in the same cycle
    // would attempt two FILL writes to the same gather buffer, and the
    // second would collide on the shared write port and silently drop.
    for (uint32_t i = 0; i < 4 * MEM_LATENCY + WARP_SIZE; ++i) {
        f.mem_if.evaluate();
        f.cache.evaluate();
        f.gather_file.commit();
        if (f.gather_file.buffer(0).filled_count == WARP_SIZE) break;
    }
    REQUIRE(f.gather_file.buffer(0).filled_count == WARP_SIZE);
    REQUIRE(f.gather_file.has_result());

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

    f.coal.evaluate();
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

    // Drive the coalescing unit to completion, ticking memory so fills can
    // land and free MSHRs for further serialized requests. Loop runs while
    // there is still work anywhere in the chain (FIFO non-empty, coal busy,
    // or outstanding cache traffic) — not just while coal is busy, since coal
    // starts idle.
    for (uint32_t step = 0;
         step < 256 &&
         (!f.coal.is_idle() || !f.ldst.fifo_empty() || !f.cache.is_idle());
         ++step) {
        f.coal.evaluate();
        f.cache.evaluate();
        f.mem_if.evaluate();
        f.gather_file.commit();
    }

    // Stores never claim the gather buffer.
    REQUIRE_FALSE(f.gather_file.is_busy(0));
    REQUIRE_FALSE(f.gather_file.has_result());
    REQUIRE(f.stats.store_misses == WARP_SIZE);
    REQUIRE(f.stats.coalesced_requests == 0);
    REQUIRE(f.stats.serialized_requests == 1);
}
