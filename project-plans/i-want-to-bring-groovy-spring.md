# Plan: Integrate DRAMSim3 as External DRAM Model

## Context

The performance simulator currently models external memory as a fixed-latency
queue (`ExternalMemoryInterface`, `sim/include/gpu_sim/timing/memory_interface.h`):
every read or write completes exactly `external_memory_latency_cycles` (default
100) after submission. This ignores command/address bus contention, bank
conflicts, refresh, row-buffer locality, and the chunked nature of cache-line
transfers over a narrow DDR bus — all first-order effects for an LLM-inference
workload that streams large weight tensors through the L1.

We want a representative DDR3 model on the DE-10 Nano target so performance
benchmarks reflect realistic bandwidth-limited behavior. DRAMSim3 is the right
granularity: cycle-accurate enough to catch bank/bus contention, fast enough
for our regression workloads, and has a clean callback-based C++ API.

**Backend selection policy:**
- **Timing-correctness tests** (Catch2 unit tests, ISA tests) keep using the
  fixed-latency backend — predictable cycle counts make oracle comparisons
  feasible. This is the **default** so existing tests are untouched.
- **Performance benchmarks** (`tests/run_workload_benchmarks.sh`,
  `tools/bench_compare.py`) opt in to the DRAMSim3 backend.

The integration must keep the existing `L1Cache` ↔ memory interface surface
(`submit_read`, `submit_write`, `has_response`, `get_response`, `evaluate`,
`commit`, `is_idle`) **unchanged at the call site** — the cache code in
`sim/src/timing/cache.cpp` (lines 112, 190, 274, 376) does not need to know
which backend is in use.

## Architecture

### Clock-domain crossing (CDC)

DRAMSim3 ticks at the DRAM data clock; the rest of the simulator ticks at the
FPGA fabric clock. Both clocks are configurable. Crossing is modeled with two
FIFOs — same as the real bridge on Cyclone V — owned by `DRAMSim3Memory`:

```
       fabric clock                         DRAM clock
   submit_read/write()  ──►  request FIFO  ──►  AddTransaction()
                                                       │
                                                       ▼
                                                  ClockTick() ×N
                                                       │
   get_response()       ◄──  response FIFO ◄──  read/write cb
```

- **Request FIFO** (bounded; `dramsim3_request_fifo_depth`, default 16):
  - Producer: `submit_read`/`submit_write` on the fabric clock.
  - Consumer: `evaluate()` drains as many entries as `WillAcceptTransaction`
    permits, calling `AddTransaction` per chunk.
  - When the FIFO is full, `submit_*` returns `false`; the cache treats this
    like any other backpressure (existing call sites check the bool already
    via the queue length, but submit_read currently returns void on the
    primary path — we'll preserve that by sizing the FIFO generously and
    asserting it isn't full when called; cache MSHR pressure already bounds
    in-flight requests).
  - Each entry carries `{line_addr, mshr_id, is_write}`.

- **Response FIFO**:
  - Producer: DRAMSim3 read/write completion callbacks, in DRAM-clock order.
  - Consumer: `get_response()` on the fabric clock.
  - Drained one entry per `get_response()` call, matching today's semantics.

### Chunked cache-line transfers

The DE-10 Nano DDR3 bus is 32 bits wide. With BL8, one DRAMSim3 transaction
moves 32 bytes. A 128-byte cache line therefore requires **4 transactions per
line**, both for reads and (write-allocate-driven) writes.

`DRAMSim3Memory` maintains a **per-MSHR reassembly buffer**:

```cpp
struct LineAssembly {
    uint32_t line_addr;
    uint32_t mshr_id;
    bool     is_write;
    uint8_t  chunks_remaining;  // initialized to chunks_per_line
};
std::unordered_map<uint64_t /*chunk_byte_addr*/, uint32_t /*mshr_id*/> chunk_to_mshr_;
std::array<LineAssembly, MAX_MSHRS> assembly_;
```

Per `submit_read(line_addr, mshr_id)`:
1. Allocate `assembly_[mshr_id]` with `chunks_remaining = chunks_per_line` (4 by default).
2. Push 4 entries into the request FIFO, one per 32-byte chunk
   (chunk_byte_addr = line_addr × line_size + i × bytes_per_burst), all tagged
   with the same `mshr_id`.
3. Each entry, when issued to DRAMSim3, registers
   `chunk_to_mshr_[chunk_byte_addr] = mshr_id`.

Read completion callback: look up `mshr_id` from `chunk_to_mshr_`, decrement
`chunks_remaining`. When it reaches 0, push **one** `MemoryResponse` for the
mshr_id into the response FIFO. The cache sees one completion per line, exactly
as it does today.

Writes don't carry an mshr_id today (`submit_write` takes only `line_addr`).
We track them by line address with a small reference counter and emit a
single write response per line. If two writes to the same line are in flight,
they share one assembly slot keyed by line_addr — matches current `submit_write`
semantics where the cache neither tags nor distinguishes them.

### Clock-ratio handling

Each `evaluate()` (called once per fabric cycle from `timing_model.cpp:327`)
advances DRAMSim3 by `dram_per_fabric` ticks, where:

```
ticks_per_fabric = dram_clock_mhz / fabric_clock_mhz   // e.g. 400/150 ≈ 2.667
```

Use a phase accumulator to handle non-integer ratios:

```cpp
phase_ += ticks_per_fabric;
while (phase_ >= 1.0) {
    drain_request_fifo_one();   // up to one AddTransaction per DRAM tick
    mem_->ClockTick();
    phase_ -= 1.0;
}
```

This keeps the long-run tick rate exact and avoids drift over multi-billion-
cycle benchmarks.

### DDR3 config for DE-10 Nano

DE-10 Nano: 1 GB DDR3 attached to the HPS, 32-bit fabric-visible bus via the
HPS-to-FPGA SDRAM bridge, DDR3-800 (400 MHz I/O, tCK = 2.5 ns). The config
file is `sim/configs/dram/DDR3_4Gb_x16_800.ini`, derived from DRAMSim3's
stock `configs/DDR3_4Gb_x16_1600.ini` with the deltas listed in step 4 below.

## Critical files to add / modify

- `sim/include/gpu_sim/timing/memory_interface.h` — refactor to abstract base
- `sim/src/timing/memory_interface.cpp` — rename impl to `FixedLatencyMemory`
- `sim/include/gpu_sim/timing/dramsim3_memory.h` — **new**
- `sim/src/timing/dramsim3_memory.cpp` — **new**
- `sim/CMakeLists.txt` — FetchContent + link, add new `.cpp`
- `sim/include/gpu_sim/config.h`, `sim/src/config.cpp` — new fields, validation, JSON, CLI
- `sim/src/timing/timing_model.cpp` (~line 150) — backend selection
- `sim/configs/dram/DDR3_4Gb_x16_800.ini` — **new**
- `sim/tests/timing/test_dramsim3_memory.cpp` — **new**
- `sim/tests/CMakeLists.txt` — register new test
- `tests/run_workload_benchmarks.sh` (and per-bench JSONs) — default benchmarks to dramsim3
- `resources/gpu_architectural_spec.md`, `resources/perf_sim_arch.md` — docs
- `.gitignore` — `build/_deps/`, `/tmp/dramsim3/`

---

## Atomic Steps

Each step lands as its own commit. Steps **1–4** are independent and can be
done in any order; step **5** depends on 1–4; later steps depend on 5. The
suite stays green at every step (default backend = `fixed`).

### Step 1: Vendor DRAMSim3 via FetchContent (no functional integration)

**Scope:** Build-system wiring only. No simulator code references DRAMSim3 yet.

- Add `FetchContent_Declare(dramsim3 …)` and `FetchContent_MakeAvailable` to
  `sim/CMakeLists.txt`. Pin to a specific upstream SHA.
- Add `target_link_libraries(gpu_sim_lib PUBLIC dramsim3)` and DRAMSim3's
  include path (gated behind a `GPU_SIM_USE_DRAMSIM3` CMake option, default ON).
- Add `build/_deps/` to `.gitignore`.
- **Verification:** `cmake -B build && cmake --build build -j8` succeeds; the
  DRAMSim3 lib is built; full regression still passes (no behavior change).

### Step 2: Refactor `ExternalMemoryInterface` into abstract base + `FixedLatencyMemory`

**Scope:** Pure refactor, no DRAMSim3 yet.

- Promote `ExternalMemoryInterface` to a pure-virtual interface.
- Move existing implementation into a `FixedLatencyMemory` class (same file or
  split — keep simple).
- `TimingModel` holds `std::unique_ptr<ExternalMemoryInterface>` and
  constructs a `FixedLatencyMemory` unconditionally for now.
- **Verification:** `ctest --test-dir build` — full suite green, byte-identical
  cycle counts vs. before (no semantics change).

### Step 3: Add config knobs (no consumer yet)

**Scope:** `SimConfig` only. Knobs are parsed but unused.

Add to `sim/include/gpu_sim/config.h`:

```cpp
std::string memory_backend = "fixed";          // "fixed" | "dramsim3"
std::string dramsim3_config_path = "";
std::string dramsim3_output_dir = "/tmp/dramsim3";
double      fpga_clock_mhz = 150.0;
double      dram_clock_mhz = 400.0;             // DDR3-800 I/O = 400 MHz
uint32_t    dramsim3_request_fifo_depth = 16;
uint32_t    dramsim3_bytes_per_burst = 32;      // BL8 × 32-bit = 32 B
```

- Wire JSON parsing in `config.cpp` (mirror existing pattern, ~line 64–129).
- Wire CLI overrides (~line 131–182).
- Validate: `memory_backend ∈ {fixed, dramsim3}`; clocks > 0; `cache_line_size_bytes`
  must be a multiple of `dramsim3_bytes_per_burst`.
- **Verification:** `ctest` passes; new unit test reads a JSON containing the
  new fields and confirms parse/validation.

### Step 4: Create `DDR3_4Gb_x16_800.ini`

**Scope:** Config file only. Not yet referenced by code.

Take DRAMSim3's stock `configs/DDR3_4Gb_x16_1600.ini` as the base and apply
exactly these deltas (everything else stays at the stock value):

```ini
[dram_structure]
device_width   = 16     ; was 8
rows           = 32768  ; was 16384

[timing]
tCK    = 2.5     ; was 1.5
CL     = 6       ; was 10
CWL    = 5       ; was 7
tRCD   = 6       ; was 10
tRP    = 6       ; was 10
tRAS   = 15      ; was 24
tRFC   = 104     ; was 74
tREFI  = 3120    ; was 5200
tFAW   = 16      ; was 20
tWR    = 6       ; was 10
tWTR_S = 4       ; was 5
tRTP   = 4       ; was 5
tXP    = 3       ; was 5
tXS    = 108     ; was 81

[system]
bus_width     = 32    ; was 64
channel_size  = 1024  ; was 2048 (MiB) → 1 GB total
```

Save as `sim/configs/dram/DDR3_4Gb_x16_800.ini`.

- **Verification:** A small standalone helper (or first DRAMSim3 unit test in
  step 6) loads the file via `dramsim3::Config` and prints summary timings;
  visual sanity check.

### Step 5: Implement `DRAMSim3Memory` backend

**Scope:** Core integration, the bulk of the work.

- New header/source under `sim/include/gpu_sim/timing/dramsim3_memory.h` and
  `sim/src/timing/dramsim3_memory.cpp`.
- Class implements the `ExternalMemoryInterface` abstract base.
- Owns the request FIFO, response FIFO, per-MSHR `LineAssembly` array, and
  `chunk_to_mshr_` map, all as described in the Architecture section.
- Constructor accepts `(const SimConfig&, Stats&)`; loads the .ini at
  `cfg.dramsim3_config_path`; computes `chunks_per_line = cache_line_size /
  bytes_per_burst`; sets up callbacks bound to its own response-push
  routines.
- `evaluate()`: runs the phase-accumulator loop, draining the request FIFO
  one entry per DRAM tick subject to `WillAcceptTransaction`, then
  `ClockTick()`.
- `submit_read`/`submit_write`: enqueue 4 chunks into the request FIFO.
  Initialize the assembly slot.
- `get_response`/`has_response`/`is_idle`: drain the response FIFO; idle
  requires both FIFOs empty AND no in-flight assemblies.
- `reset()`: tear down and rebuild the `MemorySystem`; clear FIFOs and
  assembly state.
- **Verification:** New tests in step 6.

### Step 6: Unit tests for `DRAMSim3Memory`

**Scope:** `sim/tests/timing/test_dramsim3_memory.cpp`, registered in
`sim/tests/CMakeLists.txt`.

- Construct with the DE-10 Nano .ini.
- **Chunk reassembly:** submit a single read; assert exactly one
  `MemoryResponse` arrives, with the expected `mshr_id`, after at least
  4 DRAM ticks of the burst.
- **MSHR multiplexing:** submit 8 reads to disjoint lines with distinct
  mshr_ids; assert all 8 complete and each `mshr_id` returns exactly once.
- **Sequential-vs-strided latency:** sequential reads complete faster on
  average than reads strided to thrash banks (relative check, not absolute).
- **Clock ratio:** with `fpga=150, dram=400`, calling `evaluate()` 1000
  times produces ≈2667 DRAM ticks (within ±1).
- **Idle/reset:** after draining all responses, `is_idle()` is true; after
  `reset()`, no stale state leaks.
- **Verification:** all new tests pass; full regression still green.

### Step 7: Wire backend selection into `TimingModel`

**Scope:** `sim/src/timing/timing_model.cpp` constructor (~line 150).

- Branch on `cfg.memory_backend`; construct `FixedLatencyMemory` or
  `DRAMSim3Memory` accordingly.
- Default remains `fixed`, so the entire existing test suite is unaffected.
- **Verification:** Run the full regression with default config (green).
  Then run a single small kernel with `--memory-backend=dramsim3
  --dramsim3-config-path=sim/configs/dram/DDR3_4Gb_x16_800.ini`; confirm it
  completes, functional results match the fixed-backend run, and timing
  stats differ in the expected direction (more cycles, fewer IPC).

### Step 8: Switch performance benchmarks to DRAMSim3

**Scope:** Benchmark harness only. Test suite stays on `fixed`.

- Update `tests/run_workload_benchmarks.sh` (and any per-bench config JSONs
  it consumes) to set `memory_backend=dramsim3` and point
  `dramsim3_config_path` at the new .ini.
- Ensure `tools/bench_compare.py` invocations pick up the new defaults.
- Add a `--fixed-memory` escape hatch to the benchmark script for ad-hoc
  fixed-latency runs.
- **Verification:** `bash ./tests/run_workload_benchmarks.sh --build-dir build`
  completes; `RESULT`/`SUMMARY` lines emit; numbers are believable; baseline
  recorded with `tools/bench_compare.py --baseline HEAD~1`.

### Step 9: Documentation

**Scope:** Pure docs, no code change.

- `resources/gpu_architectural_spec.md` — memory-system section: dual-backend
  policy, CDC FIFO model, chunked transfer, DE-10 Nano reference config.
- `resources/perf_sim_arch.md` — new files (`dramsim3_memory.{h,cpp}`,
  `DDR3_4Gb_x16_800.ini`), new config knobs.
- `resources/onboarding.md` — note the test-vs-bench backend split.
- `README.md` — new CLI flags, how to run benchmarks against DDR3.
- `AGENTS.md` Project Structure — `sim/configs/dram/`.

## Out of Scope

- Modeling the HPS-to-FPGA bridge buffering / clock-crossing FIFOs at gate
  level. Our two-FIFO model is functional, not register-accurate.
- Multi-channel / multi-rank DRAM configs.
- DRAMSim3 power-model output.
- Integration into the RTL flow.
