# DDR3 / DRAMSim3 Integration — Implementation Diversions

This file logs decisions and actions taken during execution of
`project-plans/i-want-to-bring-groovy-spring.md` that were **not** explicitly
spelled out in the plan. Each entry names the step, the action, and the
rationale.

## Step 1: Vendor DRAMSim3 via FetchContent

### Pinned upstream SHA
The plan says "Pin to a specific upstream SHA" but does not name one. Pinned
to the current `HEAD` of `https://github.com/umd-memsys/DRAMsim3`:

```
29817593b3389f1337235d63cac515024ab8fd6e
```

(Resolved via `git ls-remote` at integration time.)

### `FetchContent_Populate` + `add_subdirectory(... EXCLUDE_FROM_ALL)` instead of `FetchContent_MakeAvailable`
The plan literally says `FetchContent_MakeAvailable`. Using it directly causes
DRAMSim3's auxiliary targets (`dramsim3main` standalone executable,
`dramsim3test` Catch test binary) to be pulled into the default `all` target,
so `cmake --build build` would build artifacts unrelated to our simulator on
every build.

To keep the build surface minimal and avoid mixing DRAMSim3's test binary into
our `ctest` runs, the integration uses the equivalent lower-level pattern:

```cmake
FetchContent_GetProperties(dramsim3)
if(NOT dramsim3_POPULATED)
    FetchContent_Populate(dramsim3)
    add_subdirectory(${dramsim3_SOURCE_DIR} ${dramsim3_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
```

The `dramsim3` library target is still built (transitively, via
`target_link_libraries(gpu_sim_lib PUBLIC dramsim3)`), but the auxiliary
executables are not. Net behavior matches the plan's intent (DRAMSim3 lib
available to link against) while avoiding incidental build cost.

`FetchContent_Populate` is deprecated in CMake 3.30+. The project's minimum
required CMake is 3.16, where it remains the supported API. The newer
`EXCLUDE_FROM_ALL` keyword on `FetchContent_Declare` requires CMake 3.28+ and
was therefore not used.

### `.gitignore` entries
Plan requested `build/_deps/`. Added that exact line, plus a `**/_deps/`
glob, so out-of-tree build directories created in subdirectories
(`sim/build/`, `runner/build/`, etc.) are also covered. Consistent with the
existing `**/build/` pattern in the same file.

## Step 2: Refactor `ExternalMemoryInterface` into abstract base + `FixedLatencyMemory`

### Test-fixture updates not enumerated by the plan
The plan describes the refactor at the level of `TimingModel` — promote the
interface, move the impl into `FixedLatencyMemory`, have `TimingModel`
construct `FixedLatencyMemory` via the abstract `unique_ptr`. It does not
enumerate the unit-test fixtures that construct `ExternalMemoryInterface`
directly by value, but those must change as a mechanical consequence of the
interface becoming pure-virtual.

Updated to construct `FixedLatencyMemory` (same signature: `latency, stats`):
- `sim/tests/test_cache.cpp` (CacheFixture)
- `sim/tests/test_cache_mshr_merging.cpp` (MergeFixture)
- `sim/tests/test_coalescing.cpp` (CoalFixture)
- `sim/tests/test_load_gather_buffer.cpp` (two test bodies)
- `sim/tests/test_timing_components.cpp` (one test body)

The `L1Cache` ctor still takes `ExternalMemoryInterface&`, so the cache call
site is unchanged — only the concrete type at construction differs.

### `perf_sim_arch.md` updated in this step rather than step 9
The plan batches all documentation updates into step 9. The project
`CLAUDE.md` rule, however, is that "docs ship with the code" — a stale doc
is a bug. The `memory_interface.{h,cpp}` entry in `perf_sim_arch.md`
described `ExternalMemoryInterface` as the concrete fixed-latency model.
After step 2 that is no longer accurate. Updated the entry now to describe
the abstract base + `FixedLatencyMemory` split. Step 9's documentation pass
will still cover everything else (spec, onboarding, README, AGENTS.md).

### Files kept together (not split)
The plan offers "same file or split — keep simple". Kept the abstract base
and `FixedLatencyMemory` in the existing `memory_interface.{h,cpp}` pair to
minimize churn and keep test/include relationships intact. Future backends
(e.g. `DRAMSim3Memory` in step 5) live in their own files per the plan.

## Step 3: Add config knobs (no consumer yet)

### JSON parser extended for string and double fields
The pre-existing `SimConfig::from_json` was a hand-rolled key-value reader
that only handled `uint32_t` numerics and a couple of booleans. Three of the
new knobs are strings (`memory_backend`, `dramsim3_config_path`,
`dramsim3_output_dir`) and two are doubles (`fpga_clock_mhz`,
`dram_clock_mhz`). The plan says "wire JSON parsing in `config.cpp` (mirror
existing pattern)" — to actually mirror the pattern for these types I added
two new dispatch branches above the numeric fall-through:
- A string-field branch that takes `val` verbatim (the existing de-quoting
  step already strips wrapping quotes).
- A double-field branch that runs `std::stod` and routes to either clock.

The same two branches were added to `apply_cli_overrides`. The numeric
branch was extended in place for the two new uint32 fields.

### Extra validation guards
The plan enumerates three validation rules (`memory_backend` membership,
clocks > 0, line-size multiple of burst). Added two adjacent sanity checks
that the plan does not enumerate but that protect downstream code:
- `dramsim3_request_fifo_depth >= 1` (a zero-depth FIFO breaks step 5's
  bounded queue).
- `dramsim3_bytes_per_burst > 0` (avoids divide-by-zero when step 5
  computes `chunks_per_line = cache_line_size / bytes_per_burst`).

These follow the pattern of every other size-style field in `validate()`.

### Test scope broader than literal plan wording
The plan calls for "a new unit test reads a JSON containing the new fields
and confirms parse/validation". Implemented `sim/tests/test_config.cpp`
with six cases instead of one: defaults, JSON round-trip for all six knobs,
unknown-backend rejection, non-positive-clock rejection, line/burst
incompatibility rejection, and CLI override coverage. This is broader than
the literal wording but stays within step 3's "knobs are parsed but
unused" scope.

The test uses `mkstemp` for the JSON file and pulls in `<unistd.h>` for
`::close`. Registered in `sim/tests/CMakeLists.txt`.

### `perf_sim_arch.md` updated in this step
Same rationale as step 2 (docs ship with code, per CLAUDE.md). Updated:
- The `config.h`/`config.cpp` entry now lists the seven new fields and the
  three new validation rules.
- The Catch2 test catalog adds `test_config.cpp` (6 cases) and bumps the
  total from 165 to 171.

Step 9 will still cover the architectural-spec / onboarding / README pass.

## Step 4: Create `DDR3_4Gb_x16_800.ini`

### Plan's "was" values disagree with actual stock at the pinned SHA
The plan calls out specific deltas vs. DRAMSim3's stock
`configs/DDR3_4Gb_x16_1600.ini` and prints both old and new values. At the
pinned upstream SHA `29817593b3389f1337235d63cac515024ab8fd6e`, the actual
stock values differ from what the plan claims:
- `[dram_structure].device_width`: stock is `16`, plan says "was 8"
- `[dram_structure].rows`: stock is `32768`, plan says "was 16384"
- `[timing].tCK`: stock is `1.25`, plan says "was 1.5"
- `[timing].CL`: stock is `11`, plan says "was 10"
- `[timing].CWL`: stock is `8`, plan says "was 7"
- `[timing].tRCD`, `tRP`: stock is `11`, plan says "was 10"
- `[timing].tRAS`: stock is `28`, plan says "was 24"
- `[timing].tRFC`: stock is `208`, plan says "was 74"
- `[timing].tFAW`: stock is `32`, plan says "was 20"
- `[timing].tWR`: stock is `12`, plan says "was 10"
- `[timing].tWTR_S`, `tRTP`: stock is `6`, plan says "was 5"
- `[timing].tXP`: stock is `5`, plan says "was 5" (this one matches)
- `[timing].tXS`: stock is `216`, plan says "was 81"
- `[system].bus_width`: stock is `64`, plan says "was 64" (matches)
- `[system].channel_size`: stock is `4096`, plan says "was 2048"

The **target** values the plan lists (the new column) are physically
sensible DDR3-800 parameters, so I treated those as authoritative and
applied them on top of the actual stock config. The `(was X)` annotations
in my .ini file reflect the actual stock values, not the plan's "was"
values.

### Stock 1600 uses misspelled `REFI` key — corrected to `tREFI`
DRAMSim3's loader (`configuration.cc:323`) reads
`GetInteger("timing", "tREFI", 7800)`. The stock 1600 config writes
`REFI = 6240`, which DRAMSim3 silently ignores, so the effective value at
stock is the 7800 fallback default rather than 6240. The plan's directive
`tREFI = 3120` uses the correct key spelling. The new file emits
`tREFI = 3120` and omits the misnamed `REFI` line entirely.

### Full-file emission rather than sparse delta
The plan says "everything else stays at the stock value." For some keys
DRAMSim3's hard-coded loader defaults differ from the stock-file values
(e.g. `tREFI` default `7800` vs stock `6240`-as-`REFI`-which-is-ignored).
To make "stays at the stock value" actually true under DRAMSim3's loader,
I copied the entire stock config verbatim and applied the deltas in place,
rather than producing a sparse override file that would inherit DRAMSim3's
internal defaults for unspecified keys.

### Verification: throwaway loadcheck program
The plan says "a small standalone helper... loads the file via
`dramsim3::Config` and prints summary timings; visual sanity check."
Implemented as a one-shot `g++` build at `/tmp/dramsim3_loadcheck.cpp`,
linked against the FetchContent-built `libdramsim3.so` with rpath, then
deleted after running. Used the public `dramsim3::GetMemorySystem` API
(rather than the internal `dramsim3::Config` class) since it's enough to
verify that the .ini parses and yields the expected `tCK`, bus width, and
burst length. Output:

```
tCK (ns)     = 2.5
bus_bits     = 32
burst_length = 8
queue_size   = 32
```

All match expected DDR3-800 / 32-bit-bus / BL8 values. Helper not added
to the simulator build — at this step the simulator does not depend on
DRAMSim3 at runtime, and step 6 will replace the visual check with proper
Catch2 coverage.

### Documentation updates done in this step
- `AGENTS.md` (`CLAUDE.md`) Project Structure — added
  `configs/dram/` under `/sim/`. The plan defers this to step 9 but
  CLAUDE.md says docs ship with code, and `sim/configs/dram/` is a real
  directory now.
- `resources/perf_sim_arch.md` Build System — added a runtime-config
  table row for the .ini file, and added a note on the
  `sim/CMakeLists.txt` row that it vendors DRAMSim3 via FetchContent.
  The DRAMSim3-vendoring note should have landed in step 1's doc pass; I
  missed it then and caught it here.

Step 9's documentation pass is now lighter for `sim/configs/dram/` and
the build-system note (already done) but still owns the spec, onboarding,
README, and the new files added in steps 5–7.

## Step 5: Implement `DRAMSim3Memory` backend

### `std::vector` instead of `std::array<…, MAX_MSHRS>`
The plan sketches the per-MSHR reassembly storage as
`std::array<LineAssembly, MAX_MSHRS> assembly_;`. The codebase has no
`MAX_MSHRS` compile-time constant — `num_mshrs` is a `SimConfig` field
that the cache already sizes its own MSHR file from. Used
`std::vector<ReadAssembly> read_assembly_(cfg.num_mshrs)` so the storage
and the cache's MSHR pressure share a single source of truth and so the
backend doesn't need an arbitrary fixed cap.

### Reads and writes use separate assembly types
The plan describes a single `LineAssembly` struct carrying `line_addr`,
`mshr_id`, `is_write`, and `chunks_remaining`, and notes that writes
"share one assembly slot keyed by line_addr". A single struct can't
cleanly be both indexed-by-mshr-id (for reads) and keyed-by-line_addr
(for writes) at once. Split into:

- `ReadAssembly` (per-MSHR fields: `line_addr`, `chunks_remaining`,
  `active`), stored in `std::vector<ReadAssembly> read_assembly_` indexed
  by `mshr_id`.
- `WriteAssembly` (just `chunks_remaining`), stored in
  `std::unordered_map<uint32_t, WriteAssembly> write_assembly_` keyed by
  `line_addr`.

The corresponding chunk-to-owner reverse map is also split into
`read_chunk_to_mshr_` and `write_chunk_to_line_` for the same reason.

### `is_idle()` and `in_flight_count()` derived from state
The plan's pseudo-code uses an explicit in-flight counter. With writes
collapsing into per-line slots, a single counter is awkward
(increment-per-submit but only one decrement per slot completion).
Derived both queries from existing state instead:

- `is_idle()`: request FIFO empty AND response queue empty AND no
  `read_assembly_[i].active` AND `write_assembly_` empty.
- `in_flight_count()`: count of active read slots + count of write slots
  (collapsed writes count once, matching the "single response per line"
  rule).

This avoids a third book-keeping field and removes any chance of the
counter drifting from the assembly state.

### Defensive guard on MSHR-id reuse
`submit_read` returns `false` if its target `read_assembly_[mshr_id]` is
already `active`. The cache shouldn't reuse an MSHR id while one is in
flight, but the plan didn't enumerate this guard and slot corruption
would be silent without it.

### `gpu_architectural_spec.md §5.6` updated now (not deferred to step 9)
Same rationale as steps 2–4 (docs ship with code per CLAUDE.md). Added a
"Simulator memory backends" subsection that introduces the dual-backend
`fixed`/`dramsim3` selection, the asynchronous-clock CDC FIFO model, the
chunked transfer rule, and the four effects DRAMSim3 picks up that the
fixed backend doesn't. Step 9's pass can refine wording but the spec is
no longer silent about the new backend.

### `perf_sim_arch.md` updated for the new file pair
Added a full entry for `dramsim3_memory.{h,cpp}` covering ctor behavior,
clock-ratio handling, FIFO bounding, chunked transfers, write
reassembly, `evaluate()`, `reset()`, `is_idle()`, and snapshot helpers.
Also added a cross-reference note from the existing `memory_interface`
entry pointing at both backends.

### `UNTESTED.md` entry
The plan structures impl (step 5) and tests (step 6) as separate
atomic commits. Per CLAUDE.md ("Change kept without targeted tests →
add entry to UNTESTED.md"), added an entry for `DRAMSim3Memory` flagged
high priority and to be removed when step 6 lands.

## Step 6: Unit tests for `DRAMSim3Memory`

### Test file lives flat under `sim/tests/`, not in a `timing/` subdir
The plan's path is `sim/tests/timing/test_dramsim3_memory.cpp` but every
existing test file is flat under `sim/tests/`. To avoid introducing a
half-cooked subdirectory layout (and the corresponding CMake glob/path
edits) just for one file, placed the new file at
`sim/tests/test_dramsim3_memory.cpp`. Registered through the existing
`add_gpu_test` helper.

### `.ini` path baked in via a compile definition
The test needs to load `sim/configs/dram/DDR3_4Gb_x16_800.ini`. Tests run
with their build directory as cwd, so a relative path is brittle. Passed
the absolute source-tree path through as a `target_compile_definitions`
PRIVATE macro: `DRAMSIM3_TEST_INI_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../configs/dram/DDR3_4Gb_x16_800.ini"`.
Test file `#error`s if the macro isn't defined.

### Test target gated on `GPU_SIM_USE_DRAMSIM3`
Same option that drives the FetchContent in step 1. If a downstream user
turns the option off, the DRAMSim3 backend isn't compiled in and there's
no point trying to test it.

### `dram_ticks()` snapshot helper added to `DRAMSim3Memory`
The plan's clock-ratio test ("calling `evaluate()` 1000 times produces
≈2667 DRAM ticks") needs a tick counter. Added `size_t dram_ticks_`
incremented in the phase loop and exposed via `dram_ticks()`. Reset by
`reset()`. Not enumerated by the plan; required by the literal test
spec.

### `pump_until_drained` instead of "pump until idle"
The plan's `is_idle()` includes "response queue empty". Tests submit
requests, pump until done, then drain responses from the queue. A naive
"pump until is_idle" never terminates because pumping doesn't consume
responses. Used a helper that pumps while `in_flight_count() > 0`
instead — that's the right "no more work in flight" predicate for tests
that drain responses afterward. The post-drain checks (`is_idle()`,
`response_count() == 0`) still cover the original idle invariant.

### Strided thrash pattern uses an arbitrarily large stride
The plan asks for "reads strided to thrash banks (relative check, not
absolute)". DRAMSim3's address mapping (`rochrababgco` MSB-to-LSB) means
the bank/row split depends on column count, BL, and bus width — exact
stride math is fiddly. Used `i * 8192` as the line stride (≈1 MB byte
stride). Empirically this thrashes the open-page row buffers enough that
the strided run takes ~3× as many fabric cycles as the sequential run.
Test only asserts `seq_cycles < str_cycles` (the relative check the plan
called for), not specific magnitudes.

### Per-test DRAMSim3 output dir tag
Each `make_dramsim3_config` overload takes a tag string and slots it
into `dramsim3_output_dir = "/tmp/gpu_sim_dramsim3_test_<tag>"` so
multiple test cases don't collide on the same output dir. DRAMSim3
warns to stderr when the dir doesn't exist and writes to cwd instead —
harmless for unit tests but prevents one test's stat-print artifacts
from being read by another.

### Documentation
- `resources/perf_sim_arch.md` test catalog gained
  `test_dramsim3_memory.cpp` (6 cases, total bumped 171 → 177); the
  `dramsim3_memory` snapshot-helper list updated with `dram_ticks()`.
- `UNTESTED.md` — removed the entry added in step 5; the backend is now
  covered.

### Chunk-reassembly tCK floor: tightened from `>= 4` to `>= 16`
The plan's wording for the chunk-reassembly check was "at least 4 DRAM
ticks of the burst". I implemented that literally. Caught during review
that 4 tCK is the data-bus occupancy of *one* BL8 burst, not of a full
128-byte cache line. The actual per-line floor:

- Bus width 32 bits = 4 bytes/beat; BL8 = 8 beats = 32 bytes per
  transaction. On a DDR interface (2 beats per tCK) one BL8 burst
  occupies the data bus for **8 / 2 = 4 tCK**.
- A 128-byte cache line is 4 BL8 transactions. With `tCCD_S = 4`
  back-to-back column reads pipeline tightly, so the data bus is
  occupied for **4 × 4 = 16 tCK**, full stop — physically impossible
  to be lower.
- Realistic line latency is higher: + CL = 6 (READ → first data) +
  tRCD = 6 (ACT → READ) when cold. Empirical probe against
  `DDR3_4Gb_x16_800.ini` shows **31 tCK cold (open-row first access),
  24 tCK warm (row-buffer hit)** — matches the 4 tCK + (CL + cold-row
  ACT) prediction.

Updated the chunk-reassembly test assertion to `dram_ticks() >= 16` (the
unbreakable data-bus floor; non-flaky and physically grounded). Also
added a sentence to `gpu_architectural_spec.md` §5.6 "Simulator memory
backends" spelling out the per-line math (4 tCK/burst × 4 chunks = 16
data-bus floor; ~24–31 tCK observed end-to-end). The plan's literal
wording was preserved in spirit (the test still asserts a tCK floor
related to the burst structure) but replaced with the correct value.

## Step 7: Wire backend selection into `TimingModel`

### Verification kernel and harness
The plan calls for "a single small kernel" run under both backends, then
a comparison. None of the existing benchmark binaries (`matmul_bench`,
`gemv_bench`, etc.) accept a `--memory-backend` flag yet — that surface
is owned by step 8. The runner's `gpu_sim` does pick up the new CLI
flags (via `apply_cli_overrides`, wired in step 3), but it has no way
to load the structured input matrices that the workload kernels expect,
so a runner invocation against a workload ELF would just exercise an
all-zeros memory image.

To get a real comparison without expanding step 8's scope into step 7,
wrote a one-off C++ harness at `/tmp/step7_verify.cpp` that reuses the
matmul-bench data-loading pattern inline, runs the matmul kernel under
each backend (default `fixed` and the new `dramsim3` path pointed at
`sim/configs/dram/DDR3_4Gb_x16_800.ini`), and asserts:
1. Both runs complete (`is_warp_active(w)` is false for all warps).
2. The C output matrix produced by the functional model is bit-identical
   between the two backends (the cache + memory backend do not perturb
   functional results).
3. Cycle counts differ — confirming the branch actually selects a
   different backend.

The harness was compiled against `libgpu_sim_lib.a` and the FetchContent
`libdramsim3.so`, run, and deleted (along with its DRAMSim3 output
directory) once the assertions passed. Not added to the simulator
build — step 8 will replace this ad-hoc check with a real opt-in path
through the benchmark binaries.

### Plan's "expected direction" prediction did not hold
The plan's verification clause says timing stats should differ "in the
expected direction (more cycles, fewer IPC)" under DRAMSim3. The actual
matmul measurement was **351,174 cycles fixed vs 147,288 cycles
dramsim3** — DRAMSim3 was ~2.4× *faster*, not slower.

Reason: the fixed backend hard-codes `external_memory_latency_cycles =
100` *fabric* cycles per request, while DDR3-800 row-buffer hits cost
~CL = 6 tCK ≈ 15 ns at the DRAM clock, which is ~2.25 fabric cycles at
150 MHz. Even cold accesses (tRCD + CL ≈ 12 tCK ≈ 30 ns ≈ 4.5 fabric
cycles) plus the 4-chunk burst (4 × 4 tCK ≈ 24 ns ≈ 3.6 fabric cycles)
land well under 100 fabric cycles for the matmul access pattern, which
has high row-buffer locality (tile-walks that stay within an open row
for many transactions). The 100-cycle fixed latency is a conservative
placeholder, not a tight bound on real DDR3 with locality.

The cycle counts *differing significantly* (~2.4× ratio) is itself
proof that the backend branch is live. The plan's "more cycles"
prediction would hold for a workload with poor row-buffer locality
(e.g. random scatter/gather) or a smaller `external_memory_latency_cycles`
baseline, neither of which applied here. Treated this as confirmation
that step 7 works as intended; step 8 (benchmark harness switch-over)
will likely show similar reversals on the cache-friendly kernels.

### Documentation
Per CLAUDE.md "docs ship with code", updated the `timing_model.{h,cpp}`
entry in `resources/perf_sim_arch.md` to mention that the constructor
now branches on `config.memory_backend` to select between
`FixedLatencyMemory` (default) and `DRAMSim3Memory`. The
architectural-spec entry for the dual backend was already added in
step 5 and remains accurate.

## Step 8: Switch performance benchmarks to DRAMSim3

### No "per-bench config JSONs" — backend selection lives on the bench CLI
The plan says: "Update `tests/run_workload_benchmarks.sh` (and any
per-bench config JSONs it consumes)". There are no per-bench config
JSONs in the repo; each `*_bench` binary parses its own CLI for
`--num-warps`, `--memory-latency`, `--max-cycles`, `--json`. So the
backend selection had to land on those CLIs.

Added two new flags — `--memory-backend=<fixed|dramsim3>` and
`--dramsim3-config-path=<file.ini>` — to all six bench binaries
(`matmul`, `gemv`, `fused_linear_activation`, `softmax_row`,
`embedding_gather`, `layernorm_lite`). Default for the bench binary
itself stays `--memory-backend=fixed` so a standalone invocation has
unchanged behavior; `run_workload_benchmarks.sh` and `bench_compare.py`
override to `dramsim3` by default.

### Cache backpressure bug exposed by step 8 verification
With the default `dramsim3_request_fifo_depth = 16`, the
`embedding_gather` benchmark hung at the 5,000,000-cycle limit. Root
cause: the cache's `submit_read` (cache.cpp:112, 190) and
`submit_write` (cache.cpp:376) ignore the bool return from
`ExternalMemoryInterface`. Each call pushes `chunks_per_line = 4`
entries; with 4 MSHRs and a 4-deep write buffer the worst-case in-flight
chunk count is `(4 + 4) * 4 = 32`, which doesn't fit in a 16-deep FIFO.
When the FIFO filled, `DRAMSim3Memory::submit_read` returned `false`,
the cache discarded the bool, the MSHR sat allocated forever waiting
for a fill that no DRAM transaction was in flight to satisfy.

The plan acknowledged this surface: "we'll preserve [the void return on
the cache side] by sizing the FIFO generously and asserting it isn't
full when called; cache MSHR pressure already bounds in-flight
requests." The shipped default of 16 was simply too tight for the worst
case the cache can produce. Step 8 fix:

1. Bumped `SimConfig::dramsim3_request_fifo_depth` default from 16 to
   64 in `sim/include/gpu_sim/config.h` (twice the worst-case
   conservative bound for default cache config; comfortable headroom
   for larger configs).
2. Added a `validate()` rule (gated on `memory_backend == "dramsim3"`):
   `dramsim3_request_fifo_depth >= (num_mshrs + write_buffer_depth) *
   (cache_line_size_bytes / dramsim3_bytes_per_burst)`. This converts
   the silent-drop hang into an immediate config error with an
   informative message.
3. Updated `test_config.cpp` accordingly: the default-value test now
   expects 64, and a new test case asserts that an undersized FIFO is
   rejected under `dramsim3` and accepted at the exact minimum.

The `embedding_gather` benchmark now completes in ~47k cycles under
DRAMSim3 (was hitting the 5M ceiling). Numbers across the suite are
believable: matmul/gemv/fused/softmax/layernorm all complete in tens
of thousands of cycles each.

### DRAMSim3 prints to stdout, not stderr — JSON contamination
DRAMSim3 prints `WARNING: Output directory ... not exists! Using
current directory for output!` to **stdout** (not stderr, despite what
the step-6 diversion claimed; possibly a vendor change since then).
That contaminated the bench `--json` path (`bench_compare.py` got
`json.JSONDecodeError` and silently fell back to text-output parsing).

Fixed in `DRAMSim3Memory::rebuild_memory_system`: pre-create
`cfg.dramsim3_output_dir` via `std::filesystem::create_directories`
before constructing the `MemorySystem`. Errors from `create_directories`
are intentionally swallowed — if the dir really can't be created,
DRAMSim3 falls back to cwd and emits the warning, which is the prior
behavior.

Verified by running each bench with `--json` and parsing the stdout
through `json.loads`: now succeeds without the text-fallback path.

### Plan's "expected direction" continues to fail to predict outcomes
The plan said benchmarks under DRAMSim3 should show "more cycles, fewer
IPC". The actual `bash ./tests/run_workload_benchmarks.sh` deltas
between `--fixed-memory` and the new default (dramsim3):

| bench                       | fixed cycles | dramsim3 cycles |
|-----------------------------|--------------|-----------------|
| matmul                      | 351,174      | 147,288 (0.42x) |
| gemv                        | 26,913       | 8,163 (0.30x)   |
| fused_linear_activation     | 7,169        | 2,687 (0.37x)   |
| softmax_row                 | 7,178        | 3,172 (0.44x)   |
| embedding_gather            | 184,487      | 47,222 (0.26x)  |
| layernorm_lite              | 28,823       | 14,716 (0.51x)  |

DRAMSim3 is consistently 2x–4x faster than fixed=100, even on
embedding_gather (a scatter pattern). The 100-fabric-cycle fixed
default is far more conservative than DDR3-800's open-page reality.
The plan's prediction was based on the idea that a more accurate
memory model would expose bandwidth bottlenecks the fixed model hid;
in practice the fixed model overcharges latency by enough that the
realistic model wins on every kernel in the suite. Documented this in
the bench harness's own output ("memory backend: <name>" line printed
in each summary) so the reversal is obvious to anyone reading raw
output.

### `bench_compare.py` defaults to DRAMSim3 with same-backend A/B
Updated `tools/bench_compare.py`:
- Added a `build_backend_args(args)` helper that returns the CLI flags
  to pass to each bench (defaults to `--memory-backend=dramsim3
  --dramsim3-config-path=<repo>/sim/configs/dram/DDR3_4Gb_x16_800.ini`,
  flips to `--memory-backend=fixed` under the new `--fixed-memory`
  flag).
- Threaded `backend_args` through `run_benchmark` and
  `run_all_benchmarks` so the same flags apply to both the baseline
  worktree's binaries and the current build's binaries.
- Added a "Backend: <name>" line to the comparison header so the user
  can see at a glance which model the comparison was run against.

Caveat (not fixed in step 8): if the baseline ref predates step 8, the
baseline bench binaries don't recognize `--memory-backend=` and will
fail with "unknown argument". The honest behavior in that case is for
those baseline runs to be reported as FAIL, leaving only the current
results. That matches the existing failure-handling in
`run_benchmark`. Users wanting an apples-to-apples on a pre-step-8
baseline should pass `--fixed-memory`.

### `--baseline HEAD~1` verification skipped
The plan's verification mentions running `tools/bench_compare.py
--baseline HEAD~1` to record a baseline. None of the prior plan steps
have been committed to `HEAD~1`, so there is no meaningful step-7
baseline to compare against in this workspace. Skipped the literal
invocation; verified instead that:
- `bash ./tests/run_workload_benchmarks.sh --build-dir build` completes
  with `RESULT`/`SUMMARY` lines for all 6 benchmarks, all `pass`,
  numbers in plausible ranges.
- `--fixed-memory` escape hatch produces the prior-baseline numbers.
- `python3 tools/bench_compare.py --help` parses cleanly and lists the
  new `--fixed-memory` flag.
- `bench_compare.run_benchmark` invoked directly against the current
  build picks up the new defaults and successfully `json.loads`'s the
  output (no text-fallback warning).

### Documentation
Per CLAUDE.md "docs ship with code":
- `resources/perf_sim_arch.md`: extended the `validate()` description
  to include the new FIFO-depth rule; updated the `DRAMSim3Memory`
  ctor description to mention `create_directories`; bumped
  `test_config.cpp` case count 6 → 7 and totals 177 → 178.

The bench-CLI flags and the `run_workload_benchmarks.sh
--fixed-memory` escape hatch are user-facing surface that the plan
defers to step 9 (README + onboarding pass). Not duplicated here.

## Step 8 follow-up: Tighten FIFO sizing, bound the response queue, fix cache backpressure

User feedback after step 8 landed: "Sizing the request fifo for the worst
case makes sense. I think it should be exactly equal to the worst case
number of outstanding requests since having any more is wasteful. Is the
fifo for holding responses from DRAM appropriately sized so that it can
never overflow?"

### Tightened request FIFO default 64 → 32 (= exact worst case)
The step-8 fix bumped the default to 64 (twice the worst case) for
"safety headroom". User correctly pointed out that headroom is
wasteful. Tightened to exactly `(num_mshrs + write_buffer_depth) *
chunks_per_line = 32` for the default config. `validate()` already
enforced this as the lower bound; the change just brings the *default*
down to match. Updated `test_config.cpp` accordingly and added a
test case that derives the expected value from `SimConfig` defaults
to keep the relationship explicit if any underlying default later
changes.

### Reserved-region FIFO design (the real fix)
The simple "FIFO = 32" sizing turned out to be insufficient on its
own. A fresh stress test driving peak read+write production caught
`submit_read` returning `false` 11 times under load. Root cause: a
single-pool FIFO can fill entirely with writes (8 writes' worth = 32
chunks) and starve a fresh read submission, even though only
`num_mshrs` reads can ever be in flight. The cache call sites for
`submit_read` (`cache.cpp:112,190`) ignore the bool return — the MSHR
is already allocated by then — so an overflow would leave the MSHR
allocated forever waiting for a fill that no DRAM transaction was
ever issued for.

The fix splits the FIFO into two reserved regions:
- **Read region** (`num_mshrs * chunks_per_line` slots): never
  starved, guaranteeing `submit_read` always succeeds. By
  construction at most `num_mshrs` reads are in flight at once, so
  this many slots always fit.
- **Write region** (`write_buffer_depth * chunks_per_line` slots):
  `submit_write` returns `false` when this region fills.
  `L1Cache::drain_write_buffer` now respects the bool and pops only
  on success — without that, writes leak into a silently-overflowing
  FIFO under sustained store traffic.

Implementation: `DRAMSim3Memory` tracks `write_chunks_in_fifo_` as a
counter (incremented in `submit_write`, decremented when a write
chunk is ingested by DRAMSim3 in `evaluate`). `submit_write` checks
against `write_region_capacity_`; `submit_read` checks against the
full `request_fifo_depth_`. Both push sites carry asserts that
convert any bound violation into an immediate failure rather than a
silent drop.

### Bounded response queue + push-site asserts
The response queue was a `std::deque` with no documented bound. The
worst-case depth is `num_mshrs + write_buffer_depth + chunks_per_line`:
- `num_mshrs` long-running read responses (cache drains at most one
  per cycle in `L1Cache::handle_responses`).
- `write_buffer_depth` writes' worth of finalizations that may sit
  in the queue between `mem_if_->evaluate` and the next
  `cache_->evaluate` (cache drains all writes per cycle, but they
  accumulate during a single `evaluate` if multiple chunks finalize
  simultaneously).
- A `chunks_per_line` cushion for in-evaluate completions where
  multiple slots happen to retire in the same DRAM-tick burst.

Stored as `response_queue_capacity_` in the backend; `on_read_complete`
and `on_write_complete` assert against it before pushing. Tests
expose `max_observed_response_queue()` and `response_queue_capacity()`
so the stress test can verify the runtime bound was respected.

### Cache contract change: `drain_write_buffer` must respect bool
`L1Cache::drain_write_buffer` previously popped from `write_buffer_`
unconditionally. With a bounded write region, that's a silent drop
when the region is full. Changed to:
```cpp
if (mem_if_.submit_write(write_buffer_.front())) {
    write_buffer_.pop_front();
}
```
This couples cache-side write pressure to mem-backend acceptance:
when DRAMSim3 can't ingest fast enough, the cache's write_buffer
fills up, and the cache stalls store-issue (the existing
`WRITE_BUFFER_FULL` stall path). The architectural backpressure now
flows end-to-end from DRAM scheduling into the warp scheduler.

This is a real semantic change — previously the write buffer always
drained at exactly 1 entry/cycle; now it pauses when the backend is
saturated. The fixed-latency backend never returns false on
`submit_write`, so its behavior is unchanged. Verified by re-running
the full benchmark suite under the dramsim3 backend: cycle counts
match step-8's numbers exactly (matmul=147288, gemv=8163, etc.) —
the prior FIFO sizing didn't actually exercise the overflow, so the
new strict path produces the same trace for these workloads.

### New stress test: `worst-case cache traffic never drops requests`
Added in `test_dramsim3_memory.cpp`. Drives the backend at peak
production for hundreds of cycles:
- One read per cycle into any free MSHR (distinct line addresses).
- One write per cycle drained from a wb-depth-deep model write
  buffer of distinct lines (distinct addresses defeat
  write_assembly collapsing so the bound is exercised, not bypassed).
- Cache-side drain mirrors `L1Cache::handle_responses` exactly: all
  leading writes plus at most one read per cycle.
- Write drain respects `submit_write`'s bool (pops only on success).

Asserts:
- All 256 reads and 256 writes complete (no silent drops).
- `read_submit_rejected == 0` (architectural promise: reads never
  refused).
- `write_submit_rejected > 0` (sanity: the test actually pushes the
  write region to saturation; otherwise we'd be testing the easy
  path).
- `max_observed_response_queue() <= response_queue_capacity()`.
- `is_idle()` at the end with empty FIFO and zero in-flight count.

### Documentation as architectural requirements
Per CLAUDE.md "docs ship with code", made these constraints
first-class architectural requirements in the spec:
- `gpu_architectural_spec.md §5.6` "Bounded request FIFO" effect
  rewritten to spell out the read/write region split, the cache's
  obligation to respect `submit_write`'s bool, the asserts at push
  sites, and the response-queue bound (also asserted). The "rare in
  practice" hand-wave from the original step-5 wording was wrong and
  has been replaced with the precise constraint.
- `perf_sim_arch.md` `validate()` description now spells out that
  the minimum is also the recommended value (= exact worst case),
  with sizing larger called out as wasteful and sizing smaller
  rejected. The `DRAMSim3Memory` bullet list adds the read/write
  region split, the response-queue capacity bound, and the new
  snapshot helpers. The `L1Cache::drain_write_buffer` entry now
  spells out the bool-respect contract and why a silent pop would
  lose the write.
- Test catalog updated: `test_config.cpp` 7 → 8 cases (added the
  default-equals-worst-case derivation test);
  `test_dramsim3_memory.cpp` 6 → 7 (added the stress test); totals
  178 → 180.

## Backpressure-propagation review and integration test

User question after the FIFO-sizing follow-up landed: "Is the
backpressure write stall being properly passed up so earlier
hardware blocks also stall?"

### Code-trace audit
Walked the chain end-to-end:
1. `DRAMSim3Memory::submit_write` → `false` when the write region
   is full.
2. `L1Cache::drain_write_buffer` (cache.cpp:373) — fixed in the
   FIFO-sizing follow-up to pop only on success. `write_buffer_`
   stays at `write_buffer_depth`.
3. `L1Cache::process_store` (cache.cpp:126,230) — sees write
   buffer full, sets `stalled_=true` and
   `stall_reason_=WRITE_BUFFER_FULL`, increments
   `write_buffer_stall_cycles`. (Existing code; covered by
   `test_cache.cpp:246`.)
4. `CoalescingUnit::evaluate` (coalescing_unit.cpp:12) early-returns
   on `cache_.is_stalled()`; if the stall sets mid-cycle, the
   `accepted=false` paths at lines 75 and 100 keep `processing_`
   and the buffered entry, so the LD/ST FIFO doesn't drain.
5. `LdStUnit::evaluate` (ldst_unit.cpp:31) keeps `busy_=true` when
   it can't push `pending_entry_` into a full `addr_gen_fifo_`.
6. `LdStUnit::is_ready()` returns `!busy_` → false.
7. `WarpScheduler` `unit_ready_fn` (timing_model.cpp:172) returns
   false for LDST → `SchedulerIssueOutcome::UNIT_BUSY_LDST` → warp
   gets `WAIT_UNIT_LDST`.
8. The warp currently inside coalescing maps to `AT_REST` /
   `WAIT_L1_WRITE_BUFFER` in `build_cycle_snapshot`
   (timing_model.cpp:597-599).

Conclusion: structurally sound. Each stage is a single bool/state
forward at a known line; the only piece that was historically
missing — `drain_write_buffer` ignoring its return — was fixed in
the FIFO-sizing follow-up.

### Why existing benchmarks don't trip the stall
Confirmed empirically: every workload in the suite reports
`write_buffer_stall_cycles = 0` under DRAMSim3. DDR3-800 retires
writes faster than these kernels produce them, so the saturation
point isn't reached. The chain works; it just doesn't fire on
balanced workloads.

### Integration test added
Per user request after the audit, added an integration-level
Catch2 case at the bottom of `sim/tests/test_dramsim3_memory.cpp`:
`DRAMSim3Memory + L1Cache: write-region saturation propagates to
cache stall`. Wires `L1Cache` directly to `DRAMSim3Memory` with
`write_buffer_depth=1` so saturation is reachable in a few cycles,
pre-installs a cache line, then issues 64 store hits while ticking
the cache + memory + drain together each cycle. Asserts:
- `process_store` is rejected at least once with
  `cache.is_stalled()` and `stall_reason() == WRITE_BUFFER_FULL`.
- `write_buffer_stall_cycles > 0` and covers every rejection.
- `external_memory_writes == 64` (no silent drops — the
  bool-respecting drain held entries when `submit_write` failed).
- After full drain: `mem.is_idle()` and
  `cache.write_buffer_size() == 0` (no leaks).

Test docstring documents what's covered here vs. what's covered
by code review + `test_cache.cpp:246` for the upstream
coalescing → LDST → scheduler → trace mapping (each stage is a
single bool/conditional and changes rarely).

### Documentation
- `perf_sim_arch.md` test catalog: `test_dramsim3_memory.cpp`
  bumped 7 → 8; totals 180 → 181. The new case description
  spells out exactly which step in the chain is exercised
  ("`submit_write` rejection causes `drain_write_buffer` to leave
  entries in place, `write_buffer_` fills, `process_store` stalls
  with `WRITE_BUFFER_FULL`, `write_buffer_stall_cycles` increments,
  and no writes are silently lost") so a future reader can see at
  a glance that this is the integration test, not just another
  unit case.

## Step 9: Documentation

### What was already done
Most of step 9's surface had been written incrementally per CLAUDE.md
"docs ship with code" as steps 1–8 landed:
- `gpu_architectural_spec.md` §5.6 — dual-backend policy, CDC FIFO
  model, chunked transfer, DE-10 Nano reference config (added in
  step 5; refined in the FIFO-sizing follow-up to spell out the
  reserved-region split and response-queue bound).
- `perf_sim_arch.md` — `dramsim3_memory.{h,cpp}`, `DDR3_4Gb_x16_800.ini`,
  all new `SimConfig` knobs, validation rules, and the
  `TimingModel`/`L1Cache::drain_write_buffer` updates (steps 3–8).
- `AGENTS.md` (`CLAUDE.md`) Project Structure — `configs/dram/`
  (added in step 4).

Step 9's residual scope was therefore just `README.md` and
`onboarding.md`, plus a verification pass to make sure the spec
hadn't drifted.

### `README.md` — bench/runner CLI surface
Added three localised paragraphs:
- "Architecture At A Glance" → memory system bullet now mentions
  the dual-backend split and points at the DE-10 Nano `.ini`.
- "Workload Benchmarks" → the script's `--fixed-memory` escape
  hatch, plus a `bench_compare.py --baseline HEAD~1` example with
  the same flag mirrored. Notes that DRAMSim3 is the default for
  benchmarks.
- "Configuration And Tracing" → added `--memory-backend=` and
  `--dramsim3-config-path=` to the runner CLI list.

### `resources/onboarding.md` — test-vs-bench backend split
Added a single bullet to the "Memory system" section spelling out
the rule directly: tests stay on `fixed` (predictable cycle
counts), benchmarks default to `dramsim3` (DRAM-aware numbers).
Cross-referenced `gpu_architectural_spec.md` §5.6 for architectural
detail rather than duplicating the FIFO-model description. Also
added a one-paragraph note to the "Workload benchmarks" section
explaining `--fixed-memory` and warning against using DRAMSim3 as
a unit-test oracle (cycle counts depend on row-buffer state).

### Deliberately not duplicated
- The CDC FIFO model, chunked-transfer math, reserved-region split,
  and response-queue bound live in the spec (§5.6). README and
  onboarding link there rather than re-stating.
- The `--num-warps`/`--memory-latency`/`--max-cycles` per-bench
  CLI flags were already on the bench help text; not duplicated
  in README beyond the new memory-backend additions.
- `bench_compare.py` already had a brief reference in `CLAUDE.md`;
  the new README paragraph is the only user-facing reference,
  short and link-free since the tool's own `--help` is the
  authoritative reference.

### Verification pass — spec hadn't drifted
Confirmed that `gpu_architectural_spec.md` §5.6, `perf_sim_arch.md`
(config knobs, source-file entries, test catalog), and `AGENTS.md`
Project Structure were already in sync with the implementation
through the previous steps' incremental updates. No corrective
edits were needed; step 9's residual scope was just the README +
onboarding additions above.

## Final cleanup: `GPU_SIM_USE_DRAMSIM3=OFF` build was broken

Auditing the diversion file at completion turned up a real
inconsistency between the documented design and the actual build:

- Step 1 introduced the `GPU_SIM_USE_DRAMSIM3` CMake option, "default
  ON", as the toggle for whether DRAMSim3 is vendored and linked.
- Step 6's diversion claims "If a downstream user turns the option
  off, the DRAMSim3 backend isn't compiled in and there's no point
  trying to test it" — implying the option works.
- But step 5 added `src/timing/dramsim3_memory.cpp` to the
  *unconditional* source list in `sim/CMakeLists.txt`. That file
  `#include`s `dramsim3.h`, which only resolves when the FetchContent
  block runs (i.e. when the option is ON). With the option OFF the
  library failed to compile.

Verified by configuring with `-DGPU_SIM_USE_DRAMSIM3=OFF`: build
fails with `fatal error: dramsim3.h: No such file or directory`.

### Fix
- `sim/CMakeLists.txt`: moved `src/timing/dramsim3_memory.cpp` out
  of the unconditional list and into the `if(GPU_SIM_USE_DRAMSIM3)`
  block via `target_sources(gpu_sim_lib PRIVATE …)`. Added a public
  compile definition `GPU_SIM_USE_DRAMSIM3=1` inside the same block.
- `sim/src/timing/timing_model.cpp`: wrapped both the
  `#include "gpu_sim/timing/dramsim3_memory.h"` and the dramsim3
  branch of the backend-selection switch in `#ifdef GPU_SIM_USE_DRAMSIM3`.
  The `#else` arm of the dramsim3 case throws
  `std::invalid_argument` with a message that names the missing
  build flag, so a runtime attempt to use the disabled backend
  fails with a clear error rather than a link error or undefined
  behaviour.
- `dramsim3_memory.h` already used a forward declaration of
  `dramsim3::MemorySystem` rather than including `dramsim3.h`, so
  the header itself stays buildable in both configurations and no
  changes were needed there.

### Verification
- `cmake -B build && cmake --build build -j8` (default ON):
  21/21 ctest pass; benchmark suite emits identical cycle counts
  to before the change.
- `cmake -B /tmp/b-off -DGPU_SIM_USE_DRAMSIM3=OFF && cmake --build /tmp/b-off -j8`:
  builds clean, 20/20 ctest pass (the test target
  `test_dramsim3_memory` is already gated on the same option in
  `sim/tests/CMakeLists.txt`, so it's correctly excluded under OFF).
- The 20-test count under OFF excludes only `test_dramsim3_memory`,
  matching the test-catalog gating already documented in step 6.

### Why this lasted until now
None of the prior steps' verification ran with the option flipped
to OFF — every diversion entry implicitly assumed the default ON
build. The broken-OFF state didn't fail any green-CI gate, only the
CMake-option contract that the code claimed to support. Caught by
re-reading the diversion file end-to-end looking for unfulfilled
forward references.
