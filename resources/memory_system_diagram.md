# Memory & Cache System Diagram

Block diagram of the SM memory subsystem and the hardware around it. Cross-references are to `gpu_architectural_spec.md`.

## Data path (loads & stores)

```
                      ┌─────────────────────────────────────────────────┐
                      │          LD/ST Units  (8 lanes, §5.1)           │
                      │   per-warp address generation: base + offset    │
                      └──────────────────────┬──────────────────────────┘
                                             │ {warp_id, dest_reg, op,
                                             │  32 addrs, store_data}
                                             ▼
                                ┌──────────────────────────┐
                                │   AGEN → Cache FIFO      │
                                │  (decouples agen rate    │
                                │   from cache rate, §5.1) │
                                └──────────────┬───────────┘
                                               │
                                               ▼
                              ┌───────────────────────────────┐
                              │     Coalescing Unit  (§5.2)    │
                              │  Phase 1: all-or-nothing       │
                              │  blocking per-warp; falls back │
                              │  to 32 serial requests         │
                              └──┬─────────────────────────┬───┘
            claim gather buffer  │                         │ store path
            (loads only)         │                         │ (no gather buffer,
                                 │                         │  no writeback)
                                 ▼                         │
        ┌───────────────────────────────────────┐          │
        │  Load Gather Buffers  (§5.2.1)        │          │
        │  one per resident warp                │          │
        │  fields: busy, dest_reg, values[32],  │          │
        │  slot_valid[32], filled_count         │          │
        │  one write port (hit vs fill arbitrate)│         │
        └───────┬──────────────────▲────────────┘          │
                │ filled_count==32 │ per-lane writes       │
                │ → writeback req  │ (hit path / fill path)│
                ▼                  │                       │
   ┌──────────────────────┐        │                       │
   │  Writeback Arbiter   │        │                       │
   │  (§4.7, §5.5)        │        │                       │
   │  round-robin among   │        │                       │
   │  ALU / MUL / DIV /   │        │                       │
   │  TLOOKUP / MSHR-fill │        │                       │
   └─────────┬────────────┘        │                       │
             │ 32 values, 1 cycle  │                       │
             ▼                     │                       │
       ┌──────────┐                │                       │
       │ RegFile  │                │                       │
       └──────────┘                │                       │
                                   │                       │
                ┌──────────────────┴───────────────────────┴─────┐
                │           L1 Data Cache  (§5.3)                │
                │  direct-mapped, write-through, write-allocate  │
                │  128-B lines, parameterizable size (default    │
                │  16 KB)                                         │
                │                                                 │
                │   ┌─────────────────────────────────────────┐  │
                │   │ Tag/Data Arrays (1 access/cycle)        │  │
                │   └─────────────────────────────────────────┘  │
                │                                                 │
                │   ┌──────── MSHRs (default 4, §5.3.1) ───────┐ │
                │   │ cache_line_addr, lane_mask[32],          │ │
                │   │ byte_offsets[32], dest_reg, warp_id,     │ │
                │   │ next_in_chain, is_secondary,             │ │
                │   │ store_data, store_byte_en                │ │
                │   │ → secondary chain merges same-line misses│ │
                │   └──────────────────────────────────────────┘ │
                │                                                 │
                │   ┌──────── Write Buffer (FIFO, §5.4) ───────┐ │
                │   │ entries = (line_addr, 128-B line)        │ │
                │   │ drains write-through traffic             │ │
                │   └──────────────────────────────────────────┘ │
                └────────────┬───────────────────────────▲────────┘
                             │ miss reads                │ fill returns
                             │ + write-buffer drains     │ (≤1 line/cycle)
                             ▼                           │
                   ┌─────────────────────────────────────┴──┐
                   │       Memory Interface  (§5.6)          │
                   │  simple req/ready + resp/valid handshake│
                   │  cache-line granularity                 │
                   └────────────────────┬───────────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │   mem_bus_wrapper        │
                          │  Avalon-MM (Phase 1) /   │
                          │  AXI4 (swappable)        │
                          └────────────┬─────────────┘
                                       │
                                       ▼
                          ┌──────────────────────────┐
                          │   External DRAM          │
                          │  (DDR3/DDR4 controller)  │
                          │  default 100-cycle lat., │
                          │  64 MB                    │
                          └──────────────────────────┘
```

## Adjacent on-chip storage (not behind L1)

```
   ┌────────────────────┐         ┌────────────────────┐
   │  Instruction BRAM  │         │  Lookup Table BRAM │
   │  (§4.1, §5.5)      │         │  (§3 TLOOKUP)      │
   │  preloaded by DMA  │         │  preloaded by DMA  │
   │  read by Fetch     │         │  read by TLOOKUP   │
   │                    │         │  unit (dual-port)  │
   └─────────▲──────────┘         └─────────▲──────────┘
             │                              │
             │ writes during preload        │
             └──────────────┬───────────────┘
                            │
                  ┌─────────┴──────────┐
                  │   DMA Engine §6.7  │
                  │  state machine,    │
                  │  no internal FIFO  │
                  └─────────▲──────────┘
                            │ shares the memory interface (§5.6)
                            │ before kernel launch only — no
                            │ contention with L1
                            ▼
                    (External DRAM, above)

   ┌────────────────────┐
   │  Host CSR Block    │   Avalon-MM / AXI4-Lite slave
   │  (§6, §6.1)        │   ← host CPU programs DMA, launches
   │                    │     kernel, polls status
   └────────────────────┘
```

## Key invariants

- Loads reach the register file **only** via the per-warp load gather buffer; the writeback arbiter commits 32 lanes in one cycle and clears the scoreboard pending bit for `(warp_id, dest_reg)`.
- Stores never produce a writeback and never touch a gather buffer; the coalescing unit holds the FIFO entry until the cache has accepted the final lane.
- The cache fill port accepts ≤1 line per cycle. A fill that collides with a hit-path extraction targeting the same gather buffer wins the single gather-buffer write port; the hit path stalls one cycle.
- Write-allocate: store misses fetch the line, merge the store, then push the updated line into the write buffer. A full write buffer stalls **all** MSHR fills (load-miss fills included).
- DMA and the L1 share the external memory interface but never overlap — DMA runs only before kernel launch.
