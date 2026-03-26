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
- **RISC-V ISA reference:** [/resources/RISCV_CARD.md](/resources/RISCV_CARD.md) — reference card for the RISC-V instruction set. The base ISA for this project is RV32IM (integer base + multiply/divide extension).
- **FPGA implementation notes:** [/resources/fpga_implementation_notes.md](/resources/fpga_implementation_notes.md) — BRAM usage, DSP slice packing, resource estimation, clock domains, and synthesis portability guidelines.

## Project Structure

```
/resources/
  gpu_architectural_spec.md    # Full architectural spec (the source of truth)
  fpga_implementation_notes.md # FPGA-specific implementation guidance
  RISCV_CARD.md                # RISC-V ISA reference card
```

## Design Principles

1. **Parameterize everything** — warp count, pipeline depths, cache size, execution unit widths are all top-level parameters.
2. **Decouple with FIFOs/buffers** — no single slow stage blocks the pipeline.
3. **Uniform interfaces** — all execution units use valid-in/valid-out with warp tags.
4. **Hide latency via warp switching** — multiple resident warps cover memory and long-latency operation delays.
5. **FPGA-friendly** — DSP slices for multiply, BRAMs for storage, no associative matching or multi-ported structures.

## Target Workload

Quantized LLM inference: INT8 weights × INT8 activations with INT32 accumulation for matrix operations, ROM lookup tables with software interpolation for nonlinear functions (softmax, GELU, SiLU, layer norm). Realistic demo targets are 100M–300M parameter models.