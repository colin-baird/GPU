    .option norvc

    # divide_chain: probes spec §4.6 iterative DIV (32-cycle, busy-until-
    # complete) + RV32M div-by-zero behavior (DIV/0 → -1).
    #
    # ARG0 (x1) = iteration count (chain depth).
    # ARG1 (x2) = output base address (per-lane sw at base + lane*4).
    #
    # Each iteration issues a self-dependent DIV (x10 = x10 / x11), which
    # forces the next DIV to wait on the scoreboard release of x10. After
    # the chain, one extra DIV-by-zero is performed; expected RV32M result
    # is -1. The kernel stores the div-by-zero result so bench-side
    # functional verification can confirm it.

    .section .text
    .globl _start

_start:
    mv   x12, x1            # chain depth
    mv   x20, x2            # output base
    csrr x21, 0xC01         # lane_id
    slli x22, x21, 2        # lane*4
    add  x20, x20, x22      # per-lane output address

    li   x10, 1000000       # initial dividend
    li   x11, 3             # divisor (non-zero)

loop:
    div  x10, x10, x11      # dependent: each DIV reads/writes x10
    addi x12, x12, -1
    bnez x12, loop

    li   x14, 0
    div  x15, x10, x14      # div-by-zero: expected x15 = -1

    sw   x15, 0(x20)        # store div-by-zero result

    ecall
