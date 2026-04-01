# FPGA GPU Accelerator Project

## What This Is

A from-scratch GPU design in Verilog targeting FPGA synthesis and simulation. The goal is a single streaming multiprocessor (SM) that executes SIMT warps of 32 threads using the RISC-V RV32IM ISA with custom extensions, ultimately running quantized LLM inference (INT8×INT8 → INT32 accumulate).

The design is loosely modeled on NVIDIA SM architecture but simplified for FPGA feasibility: integer-only arithmetic, no shared memory, no divergence handling, single SM.

## Architecture at a Glance

- **ISA:** RV32IM + two custom instructions (VDOT8 for packed INT8 dot-product accumulate, TLOOKUP for fast nonlinear function approximation)
- **Pipeline:** Fetch → Decode → Issue (warp scheduler) → Operand Collect → Dispatch → Execute → Writeback
- **Warps:** 32 threads per warp, 4–8 resident warps per SM (parameterizable)
- **Execution units:** ALU, pipelined multiply/VDOT8, iterative divide, LD/ST, TLOOKUP — each with independent dispatch controllers
- **Memory:** Direct-mapped L1 data cache (write-through, write-allocate), MSHRs for non-blocking misses, address coalescing, external memory via Avalon-MM behind a swappable bus wrapper
- **Host interface:** CSR register block for kernel configuration, DMA engine for loading instruction and lookup table BRAMs before launch

## Key References

- **Architectural specification:** [/resources/gpu_architectural_spec.md](/resources/gpu_architectural_spec.md) — the fully specified architectural spec covering ISA, pipeline, memory system, host interface, register file, scoreboard, and all design decisions.
- **RISC-V ISA reference:** [/resources/riscv_card.md](/resources/riscv_card.md) — reference card for the RISC-V instruction set. The base ISA for this project is RV32IM (integer base + multiply/divide extension).
- **Performance Sim Documentation:** [/resources/perf_sim_arch.md](/resources/perf_sim_arch.md) —  a single reference sheet covering every file in the simulator
- **C++ Coding Standard:** [/resources/cpp_coding_standard.md](/resources/cpp_coding_standard.md) — naming, formatting, ownership, error handling, and structural conventions for all C++ code
- **Performance Reference Methodology:** [/resources/perf_reference_methodology.md](/resources/perf_reference_methodology.md) — analytical methodology for deriving reference performance statistics from the architectural spec, including worked examples and timing derivations

## Project Structure

```
/sim/                          # Performance simulator library (C++17)
  src/                           # Functional model, timing model, decoder, etc.
  include/gpu_sim/               # Public headers
  tests/                         # Catch2 unit tests
/runner/                       # Backend router and executable
  src/                           # Entry point, backend abstraction, backends
  include/runner/                # Runner headers
/tests/                        # Backend-agnostic test suites
  riscv-isa/                     # Official RISC-V ISA compliance tests
/resources/
  gpu_architectural_spec.md      # Full architectural spec (the source of truth)
  riscv_card.md                  # RISC-V ISA reference card
  perf_sim_arch.md               # Documentation for the simulator and runner
  cpp_coding_standard.md         # C++ style and conventions for the simulator
```

Build from the project root: `cmake -B build && cmake --build build -j8`

## Design Principles

1. **Parameterize everything** — warp count, pipeline depths, cache size, execution unit widths are all top-level parameters.
2. **Decouple with FIFOs/buffers** — no single slow stage blocks the pipeline.
3. **Uniform interfaces** — all execution units use valid-in/valid-out with warp tags.
4. **Hide latency via warp switching** — multiple resident warps cover memory and long-latency operation delays.
5. **FPGA-friendly** — DSP slices for multiply, BRAMs for storage, no associative matching or multi-ported structures.

## Target Workload

Quantized LLM inference: INT8 weights × INT8 activations with INT32 accumulation for matrix operations, ROM lookup tables with software interpolation for nonlinear functions (softmax, GELU, SiLU, layer norm). Realistic demo targets are 100M–300M parameter models.

## Instructions
- When updating files in the performance simulator, be sure to update the relative entries in perf_sim_arch.md.
- Don't make assumptions about any interfaces, consult the documentation and if necessary consult the definition or implementation files.
- When creating a new documentation artifact, use Markdown, place the file in ./resources, and add a pointer to CLAUDE.md.
- Strictly adhere to the architectural spec, ask before making assumptions or deviations. 
- Strictly adhere to the coding standard documentation.
- When creating new sources of build artifacts, be sure to update the global .gitignore to exclude them from tracking.
- After a session with major changes, make a new github commit with a message describing the changes that have been made.