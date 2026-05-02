    .option norvc

    # panic_drain_test: probes spec §4.8.1 panic drain bound and the
    # "in-flight" definition.
    #
    # The kernel issues a 32-cycle DIV that will still be executing in the
    # divide unit when the subsequent EBREAK reaches decode and triggers the
    # panic controller. The panic controller's drain step (§4.8.1 step 5)
    # must wait for the DIV to retire before the SM marks all warps
    # inactive — this is exactly the worst-case drain scenario the spec
    # bounds at MAX_DRAIN_CYCLES = 32.
    #
    # ARG0 (x1) = panic cause code (written to x31 per §4.8.3 before EBREAK).

    .section .text
    .globl _start

_start:
    li   x10, 1000000       # dividend
    li   x11, 3             # divisor
    div  x10, x10, x11      # 32-cyc DIV; will be in-flight at EBREAK
    mv   x31, x1            # panic cause -> r31 (spec §4.8.3 convention)
    ebreak                  # SM halts; drain step waits for DIV
