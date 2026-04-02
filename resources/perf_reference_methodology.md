# Performance Reference Methodology

**Status:** Retired as a gating oracle.

This document used to define a Python analytical flow for generating expected timing statistics from ELF binaries. That flow is no longer treated as ground truth for the timing model.

The active validation workflow now lives in:

- [`/resources/perf_alignment_validation.md`](/resources/perf_alignment_validation.md)
- [`/resources/perf_alignment_audit_matrix.md`](/resources/perf_alignment_audit_matrix.md)

## Why The Analytical Flow Was Retired

The analytical model was useful for early bring-up because it provided an implementation-independent dynamic trace and a quick consistency check against ISA tests.

It was retired as the primary oracle for three reasons:

1. It encoded a narrower timing model than the simulator and eventually diverged from the architecture contract.
2. It omitted real timing effects that materially affect total cycles:
   - duplicate outstanding misses to the same cache line
   - writeback arbitration conflicts
   - completion waiting on memory/writeback drain
3. It encouraged re-implementing the timing model in a second codebase instead of validating the simulator’s observable contract directly.

## Current Role

`/tests/references/` is retained for historical comparison and exploratory analysis only.

- `analyze_elf.py` may still be useful for rough analytical estimates.
- `validate.py` may still be useful when investigating regressions against the old ISA-test reference set.
- Neither script is part of the primary acceptance gate for performance-model alignment.

## Replacement

The replacement is the manifest-driven alignment gate:

- microbench scenarios encoded directly in the simulator test suite
- machine-readable expectation manifests
- validation against `Stats`, committed cycle snapshots, final register state, and panic diagnostics

Run it with:

```bash
ctest --test-dir build -R test_alignment --output-on-failure
```
