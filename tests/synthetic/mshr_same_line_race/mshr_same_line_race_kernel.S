    .option norvc

    # mshr_same_line_race: probes spec §5.3.1 MSHR same-line merging.
    #
    # ARG0 (x1) = shared cache-line base (128B-aligned, all warps load this)
    #
    # All resident warps coalesce-load from the same cache line near-
    # simultaneously. The first warp to reach the cache should miss and
    # allocate a primary MSHR; the remaining warps should allocate secondary
    # MSHRs on the same line. Expected: 1 external memory read, N-1 secondary
    # allocations (mshr_merged_loads = N-1).

    .section .text
    .globl _start

_start:
    csrr x10, 0xC01         # lane_id
    slli x11, x10, 2        # lane_id * 4

    add  x12, x1, x11       # shared base + lane*4 (all warps same line)

    lw   x14, 0(x12)        # coalesced load — every warp races for same line
    xor  x15, x14, x14      # consume result

    ecall
