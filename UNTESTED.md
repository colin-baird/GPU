# Untested Changes

Changes that passed regression but have not yet received targeted test coverage. Each entry should be removed once targeted tests are written and committed, or once the change is reverted.

## Format

```
### <short description>
- **Date:** YYYY-MM-DD
- **Commit:** <short hash>
- **What changed:** <brief description of the architectural/implementation change>
- **Why untested:** <"pending user decision" | "deferred" | other reason>
- **Test priority:** <high | medium | low>
```

## Entries

### TLOOKUP intra-warp 2-lane/cycle BRAM pipelining
- **Date:** 2026-04-23
- **Commit:** (logged during TLOOKUP timing-test retrofit)
- **What changed:** The architectural spec §2.3 / §4.5 describes the TLOOKUP unit as a pipelined dual-port BRAM reading 2 lanes/cycle (16 issue + 1 drain = 17 cycles). The timing model collapses the 32 lanes into a single 17-cycle countdown with no per-lane state exposed at the unit interface.
- **Why untested:** structural gap — intra-warp per-lane progress is not observable without either instrumentation exposing per-lane state or a lower-level structural model. Only the 17-cycle aggregate warp latency is bound.
- **Test priority:** low
