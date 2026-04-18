#include "catch.hpp"
#include "gpu_sim/timing/load_gather_buffer.h"
#include "gpu_sim/timing/cache.h"
#include "gpu_sim/timing/coalescing_unit.h"
#include "gpu_sim/timing/ldst_unit.h"
#include "gpu_sim/timing/memory_interface.h"
#include "gpu_sim/stats.h"

using namespace gpu_sim;

namespace {

constexpr uint32_t NUM_WARPS = 4;
constexpr uint32_t LINE_SIZE = 128;
constexpr uint32_t CACHE_SIZE = 4096;
constexpr uint32_t NUM_MSHRS = 32;
constexpr uint32_t WB_DEPTH = 16;
constexpr uint32_t MEM_LATENCY = 4;
constexpr uint32_t FULL_MASK = 0xFFFFFFFFu;

std::array<uint32_t, WARP_SIZE> make_values(uint32_t base) {
    std::array<uint32_t, WARP_SIZE> v{};
    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        v[i] = base + i;
    }
    return v;
}

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
    input.trace.mem_addresses = addrs;
    input.trace.decoded.rd = dest_reg;
    input.trace.decoded.has_rd = true;
    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        input.trace.results[i] = 100 + i;
        input.trace.mem_size[i] = 4;
    }
    return input;
}

} // namespace

TEST_CASE("LoadGatherBuffer: coalesced write fills all 32 slots in one cycle",
          "[gather_buffer]") {
    Stats stats;
    LoadGatherBufferFile file(NUM_WARPS, stats);

    file.claim(0, /*dest_reg=*/7, /*pc=*/0, /*issue_cycle=*/0, /*raw=*/0);
    REQUIRE(file.is_busy(0));
    REQUIRE(file.buffer(0).filled_count == 0);

    auto values = make_values(42);
    REQUIRE(file.try_write(0, FULL_MASK, values,
                           LoadGatherBufferFile::GatherWriteSource::FILL));

    REQUIRE(file.buffer(0).filled_count == WARP_SIZE);
    REQUIRE(file.has_result());

    WritebackEntry wb = file.consume_result();
    REQUIRE(wb.valid);
    REQUIRE(wb.warp_id == 0);
    REQUIRE(wb.dest_reg == 7);
    REQUIRE(wb.values[0] == 42);
    REQUIRE(wb.values[31] == 73);
}

TEST_CASE("LoadGatherBuffer: partial write populates only selected slots",
          "[gather_buffer]") {
    Stats stats;
    LoadGatherBufferFile file(NUM_WARPS, stats);
    file.claim(0, 1, 0, 0, 0);

    // Mask with 8 set bits (lanes 0, 4, 7, 12, 17, 20, 25, 31).
    const uint32_t mask = (1u << 0) | (1u << 4) | (1u << 7) | (1u << 12) |
                          (1u << 17) | (1u << 20) | (1u << 25) | (1u << 31);
    auto values = make_values(1000);
    REQUIRE(file.try_write(0, mask, values,
                           LoadGatherBufferFile::GatherWriteSource::FILL));

    const auto& buf = file.buffer(0);
    REQUIRE(buf.filled_count == 8);
    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        bool expected = (mask >> i) & 1u;
        REQUIRE(buf.slot_valid[i] == expected);
        if (expected) {
            REQUIRE(buf.values[i] == 1000 + i);
        }
    }

    // Writeback must not fire yet.
    REQUIRE_FALSE(file.has_result());
}

TEST_CASE("LoadGatherBuffer: writeback withheld until all 32 slots valid",
          "[gather_buffer]") {
    // Serialized load where lane 0 hits an installed line and lanes 1-31
    // each miss into distinct lines. The writeback must not appear until
    // the last MSHR fill lands — this is the primary correctness property
    // of the gather-buffer refactor.
    Stats stats;
    ExternalMemoryInterface mem_if(MEM_LATENCY, stats);
    LoadGatherBufferFile gather_file(NUM_WARPS, stats);
    L1Cache cache(CACHE_SIZE, LINE_SIZE, NUM_MSHRS, WB_DEPTH, mem_if,
                  gather_file, stats);

    // Pre-install the line that lane 0's address will map to, via a separate
    // warp so we don't perturb warp 0's gather buffer.
    auto results = make_values(0);
    gather_file.claim(/*warp_id=*/1, 3, 0, 0, 0);
    REQUIRE(cache.process_load(0, 1, FULL_MASK, results, 1, 0, 0));
    for (uint32_t i = 0; i < MEM_LATENCY; ++i) mem_if.evaluate();
    cache.handle_responses();
    (void)gather_file.consume_result();
    gather_file.commit();

    // Now issue a serialized load on warp 0: lane 0 hits line 0, lanes 1..31
    // miss into distinct lines.
    gather_file.claim(/*warp_id=*/0, /*dest_reg=*/9, 0, 0, 0);
    auto values = make_values(500);

    // Lane 0 — hit.
    REQUIRE(cache.process_load(/*addr=*/0, 0, /*mask=*/1u, values, 1, 0, 0));
    REQUIRE(gather_file.buffer(0).filled_count == 1);
    REQUIRE_FALSE(gather_file.has_result());
    gather_file.commit();

    // Lanes 1..31 — each a miss on a distinct line.
    for (uint32_t lane = 1; lane < WARP_SIZE; ++lane) {
        uint32_t addr = static_cast<uint32_t>(lane) * LINE_SIZE * 100;
        uint32_t mask = 1u << lane;
        REQUIRE(cache.process_load(addr, 0, mask, values, 1, 0, 0));
        gather_file.commit();
    }

    // At this point 31 MSHRs are outstanding; only slot 0 is filled.
    REQUIRE(gather_file.buffer(0).filled_count == 1);
    REQUIRE_FALSE(gather_file.has_result());

    // Drain the fills one cycle at a time, verifying the writeback does not
    // fire early.
    // cache.evaluate() internally runs handle_responses(); calling both in
    // the same cycle would try two FILL writes per cycle and the second
    // would collide on the shared write port and silently drop. One call
    // per cycle only.
    uint32_t fills_completed = 0;
    for (uint32_t cycle = 0; cycle < 200 && fills_completed < WARP_SIZE - 1;
         ++cycle) {
        mem_if.evaluate();
        cache.evaluate();
        gather_file.commit();
        uint32_t filled_now = gather_file.buffer(0).filled_count;
        if (filled_now < WARP_SIZE) {
            REQUIRE_FALSE(gather_file.has_result());
        }
        fills_completed = filled_now - 1;
    }

    REQUIRE(gather_file.buffer(0).filled_count == WARP_SIZE);
    REQUIRE(gather_file.has_result());
}

TEST_CASE("LoadGatherBuffer: gather_buffer_stall_cycles increments when warp is busy",
          "[gather_buffer]") {
    // Directly force the coalescing unit to encounter a busy gather buffer
    // on the target warp. Under normal operation the scoreboard prevents
    // this — this test verifies the structural safety net.
    Stats stats;
    ExternalMemoryInterface mem_if(MEM_LATENCY, stats);
    LoadGatherBufferFile gather_file(NUM_WARPS, stats);
    L1Cache cache(CACHE_SIZE, LINE_SIZE, NUM_MSHRS, WB_DEPTH, mem_if,
                  gather_file, stats);
    LdStUnit ldst(8, 4, stats);
    CoalescingUnit coal(ldst, cache, gather_file, LINE_SIZE, stats);

    // Mark warp 0's gather buffer busy without issuing a load.
    gather_file.claim(0, 1, 0, 0, 0);
    REQUIRE(gather_file.is_busy(0));

    std::array<uint32_t, WARP_SIZE> addrs;
    for (uint32_t i = 0; i < WARP_SIZE; ++i) addrs[i] = 256 + i * 4;
    auto input = make_load_dispatch(0, 5, addrs);
    ldst.accept(input, 1);
    for (int i = 0; i < 16 && ldst.fifo_empty(); ++i) {
        ldst.evaluate();
        ldst.commit();
    }
    REQUIRE_FALSE(ldst.fifo_empty());

    coal.evaluate();
    coal.commit();

    REQUIRE(stats.gather_buffer_stall_cycles == 1);
    // FIFO entry must remain (not popped) because the buffer was busy.
    REQUIRE_FALSE(ldst.fifo_empty());
    REQUIRE(stats.coalesced_requests == 0);
    REQUIRE(stats.serialized_requests == 0);
}

TEST_CASE("LoadGatherBuffer: consume_result releases buffer and re-claim succeeds",
          "[gather_buffer]") {
    Stats stats;
    LoadGatherBufferFile file(NUM_WARPS, stats);

    file.claim(2, 11, 0, 0, 0);
    auto values = make_values(1);
    REQUIRE(file.try_write(2, FULL_MASK, values,
                           LoadGatherBufferFile::GatherWriteSource::FILL));
    REQUIRE(file.has_result());

    WritebackEntry wb = file.consume_result();
    REQUIRE(wb.valid);

    const auto& buf = file.buffer(2);
    REQUIRE_FALSE(buf.busy);
    REQUIRE(buf.filled_count == 0);
    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        REQUIRE_FALSE(buf.slot_valid[i]);
    }
    REQUIRE_FALSE(file.is_busy(2));

    // Re-claim must succeed (no assert, buffer clean).
    file.claim(2, 13, 0, 0, 0);
    REQUIRE(file.is_busy(2));
}

TEST_CASE("LoadGatherBuffer: round-robin emission across two completed buffers",
          "[gather_buffer]") {
    Stats stats;
    LoadGatherBufferFile file(NUM_WARPS, stats);

    // Complete buffers 1 and 3 in the same cycle. rr_pointer_ starts at 0,
    // so consume_result() should return warp 1 first, then warp 3.
    file.claim(1, 5, 0, 0, 0);
    file.claim(3, 7, 0, 0, 0);
    auto v1 = make_values(100);
    auto v3 = make_values(200);
    REQUIRE(file.try_write(1, FULL_MASK, v1,
                           LoadGatherBufferFile::GatherWriteSource::FILL));
    REQUIRE(file.try_write(3, FULL_MASK, v3,
                           LoadGatherBufferFile::GatherWriteSource::FILL));
    file.commit();

    REQUIRE(file.has_result());
    WritebackEntry first = file.consume_result();
    REQUIRE(first.warp_id == 1);
    REQUIRE(first.dest_reg == 5);

    REQUIRE(file.has_result());
    WritebackEntry second = file.consume_result();
    REQUIRE(second.warp_id == 3);
    REQUIRE(second.dest_reg == 7);

    REQUIRE_FALSE(file.has_result());

    // Next completed buffer after warp 3 should wrap back and prefer the
    // lowest index first: complete warp 0 and warp 2 simultaneously. After
    // the previous emission rr_pointer_ is (3 + 1) % 4 = 0, so warp 0 wins.
    file.claim(0, 9, 0, 0, 0);
    file.claim(2, 11, 0, 0, 0);
    auto v0 = make_values(300);
    auto v2 = make_values(400);
    REQUIRE(file.try_write(0, FULL_MASK, v0,
                           LoadGatherBufferFile::GatherWriteSource::FILL));
    REQUIRE(file.try_write(2, FULL_MASK, v2,
                           LoadGatherBufferFile::GatherWriteSource::FILL));
    file.commit();

    WritebackEntry third = file.consume_result();
    REQUIRE(third.warp_id == 0);
    WritebackEntry fourth = file.consume_result();
    REQUIRE(fourth.warp_id == 2);
}

// Hit-vs-fill port conflict: setting up a same-cycle HIT and FILL to the
// same buffer through the full cache/coalescing stack is too
// implementation-coupled to arrange reliably from a test. Exercise the
// primitive directly at the LoadGatherBufferFile interface instead — this
// verifies the port arbitration semantics that the cache relies on.
TEST_CASE("LoadGatherBuffer: FILL wins port over same-cycle HIT", "[gather_buffer]") {
    Stats stats;
    LoadGatherBufferFile file(NUM_WARPS, stats);
    file.claim(0, 3, 0, 0, 0);

    // FILL runs first (cache.cpp schedules handle_responses before hit-path
    // writes), so the port is consumed by the FILL. A same-cycle HIT must
    // return false and bump the port-conflict counter.
    auto fill_vals = make_values(10);
    REQUIRE(file.try_write(0, 0x1u, fill_vals,
                           LoadGatherBufferFile::GatherWriteSource::FILL));

    auto hit_vals = make_values(999);
    REQUIRE_FALSE(file.try_write(0, 0x2u, hit_vals,
                                 LoadGatherBufferFile::GatherWriteSource::HIT));
    REQUIRE(stats.gather_buffer_port_conflict_cycles == 1);

    // FILL's write should be the one that landed; HIT's must not have
    // overwritten or added to slot 1.
    const auto& buf = file.buffer(0);
    REQUIRE(buf.slot_valid[0]);
    REQUIRE(buf.values[0] == 10);
    REQUIRE_FALSE(buf.slot_valid[1]);
    REQUIRE(buf.filled_count == 1);

    // Next cycle the HIT should succeed.
    file.commit();
    REQUIRE(file.try_write(0, 0x2u, hit_vals,
                           LoadGatherBufferFile::GatherWriteSource::HIT));
    REQUIRE(buf.slot_valid[1]);
    REQUIRE(buf.values[1] == 1000);
    REQUIRE(buf.filled_count == 2);
}
