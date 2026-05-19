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
bool ready_out() const { return !busy_; }
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
class WarpScheduler : public PipelineStage, public RegisteredStage {
public:
    // Constructor
    WarpScheduler(uint32_t num_warps, WarpState* warps, Scoreboard& scoreboard,
                  FunctionalModel& func_model, Stats& stats);

    // Public interface (grouped by purpose)
    void seed_next() override;
    void evaluate() override;
    void commit() override;
    void reset() override;

    // Cross-stage / cross-cycle read of the committed slot (REGISTERED
    // forward edge to OperandCollector).
    const std::optional<IssueOutput>& current_output() const {
        return output_.current();
    }

private:
    // Private helpers
    bool is_scoreboard_clear(WarpId warp, const DecodedInstruction& d) const;

    // Dependencies (config / back-pointers) first
    uint32_t num_warps_;           // config
    WarpState* warps_;             // config (back-pointer)
    Scoreboard& scoreboard_;       // config (back-pointer)
    FunctionalModel& func_model_;  // config (back-pointer)
    Stats& stats_;                 // config (back-pointer)

    // State primitives — every state-holding member is exactly one of
    // Reg<T> (clock-edge register), RegFifo<T> (commit-disciplined FIFO),
    // or Wire<T> (combinational backward signal). Plain members must be
    // annotated `// config`, `// sim-instrumentation`, or `// scratch`.
    Reg<uint32_t> rr_pointer_;
    Reg<std::optional<IssueOutput>> output_;
};
```

Hand-rolling a raw `current_X_` / `next_X_` field pair is forbidden — the
lint (`tools/lint_timing_naming.py`) rejects it. See `reg.h` for the
primitive APIs and the discipline contract in
`/resources/timing_discipline.md` § State primitives.

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

All timing model components follow the double-buffered pattern. A
pipelined stage exposes `seed_next` / `evaluate` / `commit` / `reset`:

```cpp
class MyStage : public PipelineStage {
public:
    void seed_next();              // next_* = current_* (top of tick)
    void evaluate() override;      // pure fn of current_*; writes next_*
    void commit() override;        // current_ = next_
    void reset() override;         // Return to initial state
};
```

`seed_next()` runs for every stage at the top of `TimingModel::tick()`
before any `evaluate()`; `commit()` flips at the cycle boundary. See
"The `seed_next()` / `commit()` double-buffer rule" below.

If a stage exposes a back-pressure signal, add a `current_*()` accessor
that reads only its own committed state (a REGISTERED back-pressure
read), or a `next_*()` accessor for a transient computed this cycle and
read backward (a COMBINATIONAL back-pressure read).

Execution units extend `ExecutionUnit` (a separate hierarchy that
shares the same convention) with the seed/accept/result interface:

```cpp
class MyUnit : public ExecutionUnit {
public:
    void accept(const DispatchInput& input, uint64_t cycle);
    void seed_next() override;                  // next_* = current_*
    bool current_busy() const override;         // REGISTERED, committed
    bool current_has_result() const override;   // REGISTERED, committed
    WritebackEntry consume_result() override;    // pure read, no mutation
    ExecUnit get_type() const override;
};
```

---

## Cross-stage signaling discipline (timing model)

The timing model is a synchronous-logic model: it has exactly **two** kinds
of cross-stage edge, and they pair one-to-one with the two name prefixes.
(See `CLAUDE.md` Principle 6, "Synchronous pipeline discipline", for the
project-level statement; this section is the C++ contract.)

- **REGISTERED forward edge** (`current_*`) — the *only* flavor for forward
  data. The producer writes its `next_*` slot in `evaluate()`; `commit()`
  latches `next_* → current_*`; the consumer reads the producer's
  `current_*` only. A guaranteed 1-cycle handshake, independent of `tick()`
  sweep order. Every forward data path in the pipeline is a REGISTERED
  edge: fetch→decode, scheduler→operand-collector→unit→writeback-arbiter,
  coalescing→cache, cache→memory-interface, gather-buffer→writeback-arbiter.
  A combinational-*forward* edge — a consumer reading a producer's `next_*`
  along the data direction — is **forbidden**: it collapses a cycle of
  pipeline depth and makes correctness depend on `tick()` ordering.

- **COMBINATIONAL backward control edge** (`next_*`) — the *only*
  same-cycle classification, and it is restricted to **back-pressure and
  control**, never forward data. The downstream stage asserts a transient
  `next_*` signal during its `evaluate()`; the upstream stage reads it the
  same cycle (the evaluate sweep runs back-to-front, so the downstream
  producer runs first). These are the stall/ready and redirect signals:
  decode-busy back to fetch, the cache stall to coalescing, the
  memory-interface request-stall to the cache, the writeback stall to the
  units/operand-collector/scheduler, and the branch redirect from the ALU
  to fetch/decode. A COMBINATIONAL edge requires a call-site comment
  naming the producer, the consumer, and the tick-order dependency.

Direction is therefore not a free axis: REGISTERED edges are forward-data,
COMBINATIONAL edges are backward control. The cycle prefix is mechanically
encoded by the accessor's name (`current_*` / `next_*`) and the lint
(`tools/lint_timing_naming.py`) enforces that a `next_*` cross-module read
only ever flows backward (upstream reader, downstream producer). The
cross-module layer is part of the enforced lint: if its AST prerequisites or
compile database are unavailable, the default run fails rather than silently
passing a header-only check.

## Cross-stage accessor naming

Every public `const` accessor returning `bool`, `std::optional<…>`, or a
payload reference on a timing-model class must follow the rules below.
Lifecycle hooks (`evaluate`, `commit`, `reset`, `flush`, `seed_next`,
`accept`, `consume_result`, `add_source`, `set_*`) are exempt.

### Prefixes — one per cycle discipline

| Prefix | Cycle discipline | Read semantics |
|--------|------------------|----------------|
| `current_*()` | **REGISTERED forward** | const accessor returning the producer's `current_*` slot (committed state, flipped at end-of-cycle by `commit()`). Stable through the entire evaluate phase regardless of where in the sweep it is queried. The only flavor for forward data. |
| `next_*()` | **COMBINATIONAL backward** | const accessor returning a transient back-pressure / control signal (a stall, a redirect) asserted by the downstream stage and read same-cycle by the upstream one. Ordering-sensitive; call-site comment required. Never used for forward data. |

A REGISTERED back-pressure signal — a stall computed purely from the
consumer's own committed state — is still a `current_*()` accessor (e.g.
`OperandCollector::current_busy()`). A `next_*()` accessor is reserved for
a transient computed *this cycle* and read backward, e.g.
`WritebackArbiter::next_writeback_stall()`, `L1Cache::next_stalled()`,
`ALUUnit::next_redirect()`. There is no `ready_*()` prefix.

### Postfix design language — three shapes

| Shape | Returns | Postfix grammar | Examples |
|-------|---------|-----------------|----------|
| **State predicate** | `bool` | `<prefix>_<adjective>` | `current_busy()`, `current_idle()`, `next_stalled()`, `current_in_flight(w)`, `current_pending(w, r)`, `next_fifo_empty()` |
| **Possession predicate** | `bool` | `<prefix>_has_<noun>` | `next_has_result()`, `next_has_response()` |
| **Payload accessor** | non-`bool` | `<prefix>_<noun>` | `current_output()`, `current_pending_warp()`, `current_redirect_request()`, `next_fifo_front()` |

Rules:

- **State predicates** describe a *condition* the producer is in. Bare
  adjective phrase, no `is_*` / `has_*` filler. Multi-word adjectives
  (`in_flight`, `fifo_empty`) split on underscores.
- **Possession predicates** describe whether the producer *holds* a
  thing. Reserved for accessors that precede a `consume_*` call.
- **Payload accessors** return the thing itself. Bare noun.
- **Scope is carried by parameters, not name suffixes.**
  `current_busy(WarpId w)` not `current_busy_for_warp(w)`.

### Polarity — asserted = blocking

Every state predicate returns `true` when the *condition that prevents
forward progress* is in effect. The reader writes `if (predicate)
skip;` with no negation in the common case. Possession predicates have
the opposite polarity (`true` = ownership) because they precede a
consume; the two shapes don't conflict because their grammar is
distinct.

Inverse-polarity twins are forbidden. A class exposing `current_busy()`
must not also expose `current_ready()`; pick one and stick with it.

### Field-access shape — one per relationship type

| Lifetime | Holder type |
|----------|-------------|
| Owned | `std::unique_ptr<T>` (only `TimingModel` qualifies) |
| Mandatory at construction, never null | `T&` (constructor parameter) |
| Wired post-construction, may be null in tests | `T*` (`set_*` setter, `nullptr` default) |

Mixing references and pointers within one class for dependencies that
have the same lifetime is forbidden.

### No parameter-bound cross-stage reads

Cross-stage reads must have a statically resolvable receiver. Lambdas
parameterized over module pointers (e.g. `query_unit_ready` taking an
`ExecutionUnit*` and an override slot) and free functions taking a
module pointer to indirect a read (e.g. `read_redirect_request(override,
OperandCollector*)`) are forbidden — they hide the producer endpoint
from libclang static analysis. Inline at the call site, or move the
read onto a method on the producer module.

### REGISTERED state must be private

Every `current_*` / `next_*` field is `private`. Cross-stage observers
go through accessors so a future refactor that changes the field's
underlying shape (single slot vs. double-buffered, struct vs. POD) does
not change the AST surface the diagram extractor walks.

### The `seed_next()` / `commit()` double-buffer rule

Every pipelined stage seeds `next_* = current_*` for all of its
carry-forward state at the top of the tick — before any `evaluate()`
runs — via a `seed_next()` method, and flips `next_* → current_*` at
`commit()`. The consequence is the property the whole discipline relies
on: **`evaluate()` is a pure function of committed (`current_*`) state**
plus this-tick inputs. It writes only `next_*`; it never depends on a
mutation a prior `evaluate()` left behind.

This is what makes a stage *re-runnable*. A `commit()`-gated stall (a
stage that reads a back-pressure signal at the top of `commit()` and
skips the flip) leaves `next_*` discarded and `current_*` unchanged, so
`seed_next()` re-establishes `next_* == current_*` next tick and the
stage's `evaluate()` recomputes the identical result. The writeback
stall depends on exactly this: the five execution units, the operand
collector, and the warp scheduler all freeze cleanly because their
`evaluate()` is a pure function of committed state.

A stage with no genuine cross-cycle carry-forward state (e.g. the 1-cycle
ALU, whose execution slot is recomputed fresh each tick) implements
`seed_next()` as an empty body — the interface still carries it for
uniformity. Only fields that span a cycle boundary are seeded.

### State primitives — `Reg<T>` / `RegFifo<T>` / `Wire<T>`

The `current_*` / `next_*` / `seed_next()` / `commit()` shape above is
hand-rolled per stage. New state does **not** hand-roll it — it uses the
typed primitives in `sim/include/gpu_sim/timing/reg.h`, which make the
discipline structural:

- **`Reg<T>`** — a clock-edge register. `current()` is the committed
  read; `set_next()` / `next_mut()` stage the new value; `commit()`
  latches it; `seed()` re-establishes `next_ = current_`. The canonical
  encoding of a REGISTERED field.
- **`RegFifo<T>`** — a registered FIFO: `evaluate()` stages a push / pop
  intent, `commit()` applies pop-then-push, `seed()` is a no-op.
- **`Wire<T>`** — a COMBINATIONAL backward signal: `drive()` asserts,
  `value()` reads, `reset()` (top of the producer's `evaluate()`)
  de-asserts. No committed twin, no `commit()` — it is not a register.

A class owning registers derives `RegisteredStage` and calls
`register_state(&r1, &r2, ...)` once in its constructor body;
`seed_all()` / `commit_all()` / `reset_all()` then drive every
registered primitive, so a newly added field cannot be forgotten by
`seed_next()` / `commit()`. A `commit()`-gated stall wraps the whole
`commit_all()` — a stalled stage freezes every register together.

**Every state-holding member of a timing-model class is exactly one of
`Reg<T>`, `RegFifo<T>`, `Wire<T>`, or a plain member that is config
(const after construction) or sim-instrumentation (an observational
counter, annotated `// sim-instrumentation`).** A raw `current_*` /
`next_*` field pair is a lint error; new REGISTERED state must be a
`Reg<T>`. The accessor naming rules above still apply: a stage's public
`current_*()` / `next_*()` accessors forward to the underlying primitive.

### `Stats` increments belong in `commit()`

`Stats` counters are a non-hardware simulation artifact, not pipeline
state — they are not double-buffered. Because `evaluate()` may re-run on
a stalled cycle, **`Stats` increments must never be placed in
`evaluate()` or `accept()`.** Compute the per-cycle condition into a
plain scratch flag during `evaluate()` if needed, and apply the
increment as a `commit()`-phase side effect. A stage that skips
`commit()` on a stalled cycle then naturally counts that cycle exactly
once when it finally commits. (The warp scheduler is the documented
exception: it early-returns from `evaluate()` on a stalled cycle, so its
body never runs on a stalled cycle and its increments stay safely in
`evaluate()`.)

### Discipline overview rules

- **Any cross-stage signal in the timing model must be classified at the
  call site.** Plain mutable members read by one stage and written by
  another mid-evaluate are forbidden.
- **No combinational-forward edges.** A `next_*` cross-module read must
  flow backward (upstream reader). A forward `next_*` read collapses
  pipeline depth and is a hard violation.
- **New REGISTERED state uses the `reg.h` primitives** (`Reg<T>` /
  `RegFifo<T>` / `Wire<T>` + `RegisteredStage`); see "State primitives"
  above. Do not hand-roll a `current_*` / `next_*` field pair — the lint
  rejects it.
- **Pre-evaluate setters that latch live state from another stage**
  (e.g. `set_opcoll_free`, `set_decode_pending_warp`, `set_units_drained`,
  `set_unit_ready_fn`) are forbidden. Expose the signal as a
  `current_*()` accessor on the consumer instead.
- **Mid-tick mutations of committed state that bypass `commit()`**
  (side-channel `redirect_warp`, `invalidate_warp`, `reset()` cascades) are
  forbidden. Express the request as a REGISTERED signal and let each stage
  flush at its own commit.

`tools/lint_timing_naming.py` enforces the prefix / postfix / polarity
/ field-shape rules mechanically (report-only initially; CI-enforced
once Phase 6 of the naming-and-access-discipline plan lands).

See [`/resources/timing_discipline.md`](/resources/timing_discipline.md) for
the full per-boundary inventory, the rationale for each rule, and the
phasing of the cross-stage signaling refactor.

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
