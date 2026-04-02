    .option norvc

    .section .text
    .globl _start

_start:
    # x1 = arg0 = embedding table base
    # x2 = arg1 = token id array base
    # x3 = arg2 = output base
    # x4 = arg3 = embedding words per row
    mv x18, x1
    mv x19, x2
    mv x20, x3
    mv x21, x4

    csrr x22, 0xC00    # warp id
    csrr x23, 0xC01    # lane id
    csrr x24, 0xC02    # resident warps

    slli x25, x24, 5   # total token count = num_warps * 32
    slli x5, x22, 5
    add x5, x5, x23    # token index for this lane
    bgeu x5, x25, done

    slli x6, x5, 2
    add x6, x19, x6
    lw x7, 0(x6)       # token id

    slli x8, x21, 2    # row bytes
    mul x9, x7, x8
    add x9, x18, x9    # embedding row base

    mul x10, x5, x8
    add x10, x20, x10  # output row base

    addi x11, x0, 0

copy_loop:
    lw x12, 0(x9)
    sw x12, 0(x10)
    addi x9, x9, 4
    addi x10, x10, 4
    addi x11, x11, 1
    bltu x11, x21, copy_loop

done:
    ecall
