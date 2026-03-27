#include "catch.hpp"
#include "gpu_sim/timing/scoreboard.h"

using namespace gpu_sim;

TEST_CASE("Scoreboard: r0 never pending", "[scoreboard]") {
    Scoreboard sb;

    sb.seed_next();
    sb.set_pending(0, 0);
    sb.commit();

    REQUIRE_FALSE(sb.is_pending(0, 0));
}

TEST_CASE("Scoreboard: set and read pending", "[scoreboard]") {
    Scoreboard sb;

    sb.seed_next();
    sb.set_pending(0, 5);
    sb.commit();

    REQUIRE(sb.is_pending(0, 5));
    REQUIRE_FALSE(sb.is_pending(0, 6));
    REQUIRE_FALSE(sb.is_pending(1, 5)); // Different warp
}

TEST_CASE("Scoreboard: clear pending", "[scoreboard]") {
    Scoreboard sb;

    // Set pending
    sb.seed_next();
    sb.set_pending(0, 10);
    sb.commit();
    REQUIRE(sb.is_pending(0, 10));

    // Clear pending
    sb.seed_next();
    sb.clear_pending(0, 10);
    sb.commit();
    REQUIRE_FALSE(sb.is_pending(0, 10));
}

TEST_CASE("Scoreboard: double-buffer isolation", "[scoreboard]") {
    Scoreboard sb;

    // Cycle 1: set pending in next, but current should not see it yet
    sb.seed_next();
    sb.set_pending(0, 7);
    // Before commit, current should not show it
    REQUIRE_FALSE(sb.is_pending(0, 7));

    // After commit, current should show it
    sb.commit();
    REQUIRE(sb.is_pending(0, 7));
}

TEST_CASE("Scoreboard: set and clear in same cycle", "[scoreboard]") {
    Scoreboard sb;

    // Set reg 5 pending
    sb.seed_next();
    sb.set_pending(0, 5);
    sb.commit();
    REQUIRE(sb.is_pending(0, 5));

    // In one cycle: issue sets reg 8 pending, writeback clears reg 5
    sb.seed_next();
    sb.set_pending(0, 8);
    sb.clear_pending(0, 5);
    sb.commit();

    REQUIRE(sb.is_pending(0, 8));
    REQUIRE_FALSE(sb.is_pending(0, 5));
}

TEST_CASE("Scoreboard: multiple warps independent", "[scoreboard]") {
    Scoreboard sb;

    sb.seed_next();
    sb.set_pending(0, 5);
    sb.set_pending(1, 10);
    sb.set_pending(2, 15);
    sb.commit();

    REQUIRE(sb.is_pending(0, 5));
    REQUIRE_FALSE(sb.is_pending(0, 10));
    REQUIRE(sb.is_pending(1, 10));
    REQUIRE_FALSE(sb.is_pending(1, 5));
    REQUIRE(sb.is_pending(2, 15));
}

TEST_CASE("Scoreboard: seed_next preserves current state", "[scoreboard]") {
    Scoreboard sb;

    sb.seed_next();
    sb.set_pending(0, 3);
    sb.commit();

    // seed_next copies current to next, so pending(0,3) should survive
    sb.seed_next();
    // Don't modify next at all
    sb.commit();

    REQUIRE(sb.is_pending(0, 3));
}

TEST_CASE("Scoreboard: all registers can be pending", "[scoreboard]") {
    Scoreboard sb;

    sb.seed_next();
    for (uint8_t r = 1; r < 32; ++r) {
        sb.set_pending(0, r);
    }
    sb.commit();

    REQUIRE_FALSE(sb.is_pending(0, 0)); // r0 never pending
    for (uint8_t r = 1; r < 32; ++r) {
        REQUIRE(sb.is_pending(0, r));
    }
}

TEST_CASE("Scoreboard: reset clears everything", "[scoreboard]") {
    Scoreboard sb;

    sb.seed_next();
    sb.set_pending(0, 5);
    sb.set_pending(1, 10);
    sb.commit();

    sb.reset();

    REQUIRE_FALSE(sb.is_pending(0, 5));
    REQUIRE_FALSE(sb.is_pending(1, 10));
}
