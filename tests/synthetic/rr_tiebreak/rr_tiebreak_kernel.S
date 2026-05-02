    .option norvc

    # rr_tiebreak: probes spec §4.2/§4.3 round-robin scheduler tie-break.
    #
    # Each warp executes an identical pure-ALU loop with no memory ops and
    # no divergent control flow. Iteration count is taken from ARG0 (=r1).
    # The body is exactly 5 instructions: 3 ALU + 1 ALU loop counter + 1
    # backward conditional branch (predicted taken, correctly except on the
    # final iteration). With multiple warps in lockstep, the round-robin
    # pointer should select each warp in strict alternation, yielding
    # exactly equal per-warp instruction counts.

    .section .text
    .globl _start

_start:
    mv x12, x1          # iter count = ARG0
    li x13, 0           # accumulator
    li x14, 1           # increment

loop:
    add x13, x13, x14
    add x13, x13, x14
    xor x13, x13, x14
    addi x12, x12, -1
    bnez x12, loop

    ecall
