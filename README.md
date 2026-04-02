# FPGA GPU Accelerator

A from-scratch GPU design targeting FPGA synthesis. The goal is a single streaming multiprocessor (SM) that executes SIMT warps of 32 threads using the RISC-V RV32IM ISA with custom extensions, ultimately running quantized LLM inference.

## Architecture

- **ISA:** RV32IM + two custom instructions (VDOT8 for packed INT8 dot-product accumulate, TLOOKUP for nonlinear function approximation via ROM lookup tables)
- **Pipeline:** Fetch, Decode, Issue (warp scheduler), Operand Collect, Dispatch, Execute, Writeback
- **Execution units:** ALU, pipelined multiply/VDOT8, iterative divide, LD/ST, TLOOKUP
- **Warps:** 32 threads per warp, 4-8 resident warps per SM (parameterizable)
- **Memory:** Direct-mapped L1 data cache (write-through, write-allocate), MSHRs for non-blocking misses, address coalescing
- **Host interface:** CSR register block for kernel configuration, DMA engine for loading instruction and lookup table BRAMs

The design is loosely modeled on NVIDIA SM architecture, simplified for FPGA feasibility: integer-only arithmetic, no shared memory, no divergence handling, single SM.

## Project Structure

```
sim/                        Performance simulator library (C++17)
  src/                        Simulator source code
    functional/                 Functional execution model
    timing/                     Cycle-accurate timing model
  include/gpu_sim/            Headers
  tests/                      Catch2 unit tests

runner/                     Backend router and executable
  src/                        Entry point, backend abstraction, backend implementations
  include/runner/             Runner headers

tests/                      Backend-agnostic test suites
  riscv-isa/                  Official RISC-V ISA compliance tests (rv32ui + rv32um)

resources/                  Documentation
  gpu_architectural_spec.md   Full architectural specification (source of truth)
  perf_sim_arch.md            Simulator file reference
  perf_alignment_validation.md Performance alignment gate and manifest schema
  perf_alignment_audit_matrix.md Spec-to-model audit matrix
  cpp_coding_standard.md      C++ coding conventions
  riscv_card.md               RISC-V ISA reference card
```

## Quick Start

### Build

The top-level CMake builds both the simulator library and the runner executable:

```bash
cmake -B build
cmake --build build -j8
```

### Run a program

The `gpu_sim` executable accepts RISC-V ELF binaries or raw binary files and routes them through a backend system:

```bash
# Run with the performance simulator (default backend)
./build/runner/gpu_sim program.elf

# Functional-only mode (no timing simulation)
./build/runner/gpu_sim program.elf --functional-only

# With configuration
./build/runner/gpu_sim program.elf --config=config.json --num-warps=4

# Explicit backend selection
./build/runner/gpu_sim --backend=perf_sim program.elf
```

Run `./build/runner/gpu_sim --help` for all options.

### Run unit tests

```bash
cd build/sim
ctest --output-on-failure
```

### Run the performance alignment gate

```bash
ctest --test-dir build -R test_alignment --output-on-failure
```

### Run RISC-V ISA compliance tests

Requires a RISC-V cross-compiler (`riscv64-unknown-elf-gcc`).

```bash
# Build test ELFs (clones riscv-tests on first run)
cd tests/riscv-isa
make

# Run all tests through the backend router
./run_tests.sh

# Run against a specific backend
./run_tests.sh --backend=perf_sim
```

## Backend System

Programs are executed through a backend routing system. The runner (`runner/`) is separate from the simulator library (`sim/`), allowing different backends to be added without modifying the simulator. Each backend accepts a parsed ELF image and runs it to completion.

| Backend | Flag | Description |
|---------|------|-------------|
| `perf_sim` | `--backend=perf_sim` | Performance simulator with functional and cycle-accurate timing models |

To add a new backend, implement the `Backend` interface in `runner/` and register it in the factory.

## Target Workload

Quantized LLM inference: INT8 weights x INT8 activations with INT32 accumulation for matrix operations, ROM lookup tables with software interpolation for nonlinear functions (softmax, GELU, SiLU, layer norm). Target model size is 100M-300M parameters.

## Documentation

- [Architectural Specification](resources/gpu_architectural_spec.md) -- full design spec covering ISA, pipeline, memory, host interface, and all design decisions
- [Simulator File Reference](resources/perf_sim_arch.md) -- guide to every file in the simulator
- [Performance Alignment Validation](resources/perf_alignment_validation.md) -- manifest schema and active timing-validation workflow
- [Performance Alignment Audit Matrix](resources/perf_alignment_audit_matrix.md) -- current spec-to-model alignment status by rule
- [C++ Coding Standard](resources/cpp_coding_standard.md) -- naming, formatting, and structural conventions
- [RISC-V ISA Reference](resources/riscv_card.md) -- instruction set reference card
