    .option norvc

    # line_boundary_load: probes spec §5.2 coalescer line-boundary detection.
    #
    # ARG0 (x1) = aligned base address  (128B-aligned; lane n loads addr+4n)
    # ARG1 (x2) = straddle base address (aligned + 64; lanes 0..15 in line A,
    #             lanes 16..31 in line B)
    #
    # Each warp issues two coalesced-shape loads:
    #   1. Aligned base — all 32 addresses fall in one 128B line → coalescer
    #      should emit 1 coalesced cache request.
    #   2. Straddle base — lane addresses span 2 cache lines → coalescer
    #      should fall back to 32 serialized requests.
    # Result is XOR'd to keep both loads live.

    .section .text
    .globl _start

_start:
    csrr x10, 0xC01         # lane_id
    slli x11, x10, 2        # lane_id * 4 (byte offset)

    add  x12, x1, x11       # aligned base + lane*4
    add  x13, x2, x11       # straddle base + lane*4

    lw   x14, 0(x12)        # aligned load: expected coalesced
    lw   x15, 0(x13)        # straddle load: expected serialized
    xor  x16, x14, x15      # consume both results

    ecall
