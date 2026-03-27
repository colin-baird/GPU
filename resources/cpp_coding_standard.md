# C++ Coding Standard

This document codifies the style conventions used in the GPU simulator codebase (`sim/`). All new C++ code should follow these rules to maintain consistency.

---

## Naming

| Element | Convention | Examples |
|---------|-----------|----------|
| Classes | PascalCase | `TimingModel`, `FunctionalModel`, `WritebackArbiter` |
| Structs | PascalCase | `SimConfig`, `TraceEvent`, `WritebackEntry`, `FetchOutput` |
| Methods / free functions | snake_case | `is_warp_active()`, `execute_alu()`, `drain_write_buffer()` |
| Class member variables | snake_case with trailing `_` | `cycle_`, `func_model_`, `rr_pointer_`, `tags_` |
| Struct member variables | snake_case, no trailing `_` | `warp_id`, `branch_taken`, `num_warps` |
| Local variables / parameters | snake_case | `line_addr`, `dest_reg`, `max_cycles` |
| Compile-time constants | UPPER_SNAKE_CASE | `WARP_SIZE`, `MAX_WARPS`, `NUM_REGS` |
| Enum values | UPPER_SNAKE_CASE | `ALU`, `MULTIPLY`, `DIVIDE`, `NONE` |
| Type aliases | PascalCase | `WarpId`, `Address`, `WarpData` |
| File names | snake_case | `timing_model.h`, `warp_scheduler.cpp`, `alu_unit.h` |
| Test cases (Catch2) | `"Component: behavior description"` | `"Scoreboard: r0 never pending"` |
| Test tags (Catch2) | `[lowercase]` | `[scoreboard]`, `[integration]`, `[cache]` |

The trailing underscore on class members distinguishes private state from local variables at a glance. Struct members omit it because structs in this codebase have public fields that act more like named parameters.

---

## Formatting

**Indentation**: 4 spaces. No tabs.

**Braces**: K&R style -- opening brace on the same line as the declaration, closing brace on its own line.

```cpp
class Scoreboard {
public:
    void reset() {
        std::memset(current_, 0, sizeof(current_));
    }
};

if (tags_[set].valid && tags_[set].tag == tag) {
    stats_.cache_hits++;
} else {
    stats_.cache_misses++;
}
```

**Short bodies**: One-line bodies are allowed inline for trivial getters, guards, and no-ops:

```cpp
bool is_free() const { return !busy_; }
ExecUnit get_type() const override { return ExecUnit::ALU; }
if (reg == 0) return false;
```

**Line length**: Aim for 100 characters. Lines up to ~120 are acceptable to avoid awkward breaks. Break long initializer lists and function signatures across lines:

```cpp
L1Cache::L1Cache(uint32_t cache_size, uint32_t line_size, uint32_t num_mshrs,
                 uint32_t write_buffer_depth, ExternalMemoryInterface& mem_if, Stats& stats)
    : cache_size_(cache_size), line_size_(line_size),
      num_sets_(cache_size / line_size),
      tags_(cache_size / line_size),
      mshrs_(num_mshrs),
      write_buffer_depth_(write_buffer_depth),
      mem_if_(mem_if), stats_(stats) {}
```

**Blank lines**: One blank line between logical sections in a method, between method definitions, and around namespace braces. No blank lines between tightly related one-liner getters.

---

## File Organization

### Headers

Every header starts with `#pragma once`. No `#ifndef` guards.

Include order within a file:

1. Project headers (`"gpu_sim/..."`)
2. Standard library headers (`<vector>`, `<cstdint>`, ...)

```cpp
#pragma once

#include "gpu_sim/timing/pipeline_stage.h"
#include "gpu_sim/timing/warp_state.h"
#include "gpu_sim/timing/scoreboard.h"
#include "gpu_sim/functional/functional_model.h"
#include "gpu_sim/stats.h"
#include <optional>
#include <functional>
```

### Header-only vs split

Put the implementation in the header when the class is:
- Small and self-contained (under ~50 lines of logic)
- Performance-critical and benefits from inlining (e.g. `Scoreboard`, `FlatMemory`)
- A pure data holder with trivial methods (e.g. `InstructionBuffer`, `WarpState`)

Split into `.h` + `.cpp` when the class has:
- Complex method bodies (algorithms, state machines, multi-step logic)
- Many private helpers
- Heavy constructor logic

### Source files

Source files include their own header first, then any additional dependencies:

```cpp
#include "gpu_sim/timing/cache.h"

namespace gpu_sim {
// ...
} // namespace gpu_sim
```

---

## Namespaces

All code lives in the single namespace `gpu_sim`. ISA constants live in `gpu_sim::isa`.

```cpp
namespace gpu_sim {

class MyClass { ... };

} // namespace gpu_sim
```

Closing braces get `// namespace gpu_sim` comments.

`using namespace gpu_sim;` is permitted in `main.cpp` and test files only. Never in headers.

---

## Classes and Structs

**When to use struct vs class**: Use `struct` for passive data (configuration, event records, buffer entries) where all members are public. Use `class` for types with invariants, private state, or complex behavior.

**Member ordering in classes**:

```cpp
class WarpScheduler : public PipelineStage {
public:
    // Type aliases
    using UnitReadyFn = std::function<bool(ExecUnit)>;

    // Constructor
    WarpScheduler(uint32_t num_warps, WarpState* warps, Scoreboard& scoreboard,
                  FunctionalModel& func_model, Stats& stats);

    // Public interface (grouped by purpose)
    void evaluate() override;
    void commit() override;
    void reset() override;

    void set_opcoll_free(bool free) { opcoll_free_ = free; }
    void set_unit_ready_fn(UnitReadyFn fn) { unit_ready_fn_ = std::move(fn); }

    std::optional<IssueOutput>& output() { return next_output_; }
    const std::optional<IssueOutput>& current_output() const { return current_output_; }

private:
    // Private helpers
    bool is_scoreboard_clear(WarpId warp, const DecodedInstruction& d) const;

    // State (dependencies first, then owned state)
    uint32_t num_warps_;
    WarpState* warps_;
    Scoreboard& scoreboard_;
    FunctionalModel& func_model_;
    Stats& stats_;

    uint32_t rr_pointer_ = 0;
    bool opcoll_free_ = true;
    UnitReadyFn unit_ready_fn_;

    std::optional<IssueOutput> current_output_;
    std::optional<IssueOutput> next_output_;
};
```

**Member ordering in structs**: Group related fields. Use in-class default initializers:

```cpp
struct WritebackEntry {
    bool valid = false;
    uint32_t warp_id = 0;
    uint8_t dest_reg = 0;
    std::array<uint32_t, WARP_SIZE> values{};
    ExecUnit source_unit = ExecUnit::NONE;
    uint64_t issue_cycle = 0;
};
```

**`explicit`**: Mark single-parameter constructors `explicit`:

```cpp
explicit FlatMemory(uint32_t size_bytes);
explicit OperandCollector(Stats& stats);
```

---

## Types

**Use fixed-width integers** from `<cstdint>`: `uint8_t`, `uint16_t`, `uint32_t`, `int32_t`, `uint64_t`. Avoid bare `int`, `unsigned`, `long`.

**Use domain type aliases** where they add clarity: `WarpId`, `LaneId`, `RegIndex`, `Address`, `Word`. Use the raw `uint32_t` when the value has no specific domain meaning (loop counters, sizes, generic indices).

**`auto`**: Use sparingly. Acceptable for:
- Range-for iterators: `for (auto& entry : entries_)`
- Long return types from `std::make_unique` or similar
- When the type is obvious from context: `auto resp = mem_if_.get_response();`

Prefer explicit types when the type is not obvious from the right-hand side.

---

## Const Correctness

- Mark all read-only methods `const`.
- Pass large types by `const&`. Pass small types (`uint32_t`, `bool`, enums) by value.
- Provide both `const` and non-`const` accessors when mutation is sometimes needed:

```cpp
FunctionalRegisterFile& register_file() { return reg_file_; }
const FunctionalRegisterFile& register_file() const { return reg_file_; }
```

---

## Ownership and Memory

**`std::unique_ptr`** for all owning heap pointers. Construct with `std::make_unique`:

```cpp
alu_ = std::make_unique<ALUUnit>(stats);
```

**References** (`&`) for non-owning dependencies injected via constructor:

```cpp
ExternalMemoryInterface& mem_if_;
Stats& stats_;
```

**Raw pointers** (`*`) only for non-owning access to arrays or contiguous data managed by the caller:

```cpp
WarpState* warps_;  // Points into a vector owned by TimingModel
```

**No `shared_ptr`** in this codebase. Ownership is always singular and explicit.

**Stack allocation** for small objects embedded directly in their owner:

```cpp
Scoreboard scoreboard_;  // Owned directly by TimingModel, not heap-allocated
```

---

## Enums

Use `enum class` (scoped enums). Values are UPPER_SNAKE_CASE. Always include a `NONE` or `INVALID` sentinel as the last value:

```cpp
enum class ExecUnit : uint8_t {
    ALU,
    MULTIPLY,
    DIVIDE,
    LDST,
    TLOOKUP,
    SYSTEM,
    NONE
};
```

Specify the underlying type (`: uint8_t`) when the enum will be stored in packed structures.

---

## Error Handling

**Exceptions** for programming errors and invalid configuration detected at startup:
- `std::invalid_argument` for bad config values
- `std::out_of_range` for bounds violations
- `std::runtime_error` for file I/O failures

```cpp
if (num_warps < 1 || num_warps > 8) {
    throw std::invalid_argument("num_warps must be in [1, 8]");
}
```

**Return values** for runtime conditions that are part of normal operation (not errors):
- `bool` for accept/reject: `bool process_load(...)` returns false when MSHRs full
- `std::optional` for maybe-empty outputs: `std::optional<IssueOutput>`
- `int` returning -1 for allocation failure: `int MSHRFile::allocate()`

Catch exceptions at the top level (`main.cpp`). Do not use exceptions for flow control in the simulation hot path.

---

## Comments

Use `//` for all comments. No `/* */` blocks.

Comment intent, not mechanics. Good:

```cpp
// Seed next from current at start of cycle
scoreboard_.seed_next();
```

Unnecessary:

```cpp
// Increment i
i++;
```

Place a brief comment above non-obvious code blocks, algorithm steps, or design decisions:

```cpp
// All-or-nothing: if all 32 addresses in same 128B line -> 1 request
uint32_t first_line = current_entry_.trace.mem_addresses[0] / line_size_;
```

No Doxygen-style `/** */` doc comments. The architecture document (`perf_sim_arch.md`) and test files serve as interface documentation.

---

## STL Usage

**Preferred containers**:
- `std::vector` for dynamic arrays
- `std::array` for fixed-size arrays (especially per-lane data: `std::array<uint32_t, WARP_SIZE>`)
- `std::deque` for FIFOs and queues
- `std::optional` for values that may or may not be present
- `std::function` for callbacks

**Avoid**: `std::map`/`std::unordered_map` in the simulation hot path (associative lookups are not FPGA-friendly and the simulator avoids them to match hardware constraints). Use direct-indexed arrays instead.

Use `std::memset` / `std::memcpy` (from `<cstring>`) for bulk initialization of POD arrays when performance matters.

Prefer explicit loops over STL algorithms when the loop body involves domain-specific logic. The codebase uses simple `for` loops with explicit indices throughout.

---

## Pipeline Stage Pattern

All timing model components follow the double-buffered evaluate/commit pattern:

```cpp
class MyStage : public PipelineStage {
public:
    void evaluate() override;  // Read current_ state, compute next_ state
    void commit() override;    // current_ = next_
    void reset() override;     // Return to initial state
};
```

Execution units extend this with the accept/ready/result interface:

```cpp
class MyUnit : public ExecutionUnit {
public:
    void accept(const DispatchInput& input, uint64_t cycle);
    bool is_ready() const override;
    bool has_result() const override;
    WritebackEntry consume_result() override;
    ExecUnit get_type() const override;
};
```

---

## Tests

Tests use Catch2 v2 (single-header). Each test file includes `"catch.hpp"` and links against the `catch_main` object library.

Structure test names as `"Component: specific behavior"`. Group related tests with a shared tag:

```cpp
TEST_CASE("Scoreboard: r0 never pending", "[scoreboard]") {
    // Arrange
    Scoreboard sb;

    // Act
    sb.seed_next();
    sb.set_pending(0, 0);
    sb.commit();

    // Assert
    REQUIRE_FALSE(sb.is_pending(0, 0));
}
```

Use `REQUIRE` (fatal on failure) rather than `CHECK` (continues on failure) as the default assertion. Use `REQUIRE_FALSE` for negative assertions.

Test fixtures are plain structs with helper methods:

```cpp
struct CacheFixture {
    Stats stats;
    ExternalMemoryInterface mem_if{10, stats};
    L1Cache cache{4096, 128, 4, 4, mem_if, stats};

    void tick_mem(uint32_t cycles) { ... }
};
```

Place instruction encoding helpers as `static` functions at the top of test files that need them.
