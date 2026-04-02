    .option norvc

    .equ GEMV_COLS, 256
    .equ GEMV_PACKED_COLS, (GEMV_COLS / 4)

    .macro VDOT8 rd, rs1, rs2
        .insn r 0x0b, 0x0, 0x00, \rd, \rs1, \rs2
    .endm

    .section .text
    .globl _start

_start:
    # x1 = arg0 = packed matrix base
    # x2 = arg1 = packed input vector base
    # x3 = arg2 = output vector base
    # x4 = arg3 = row count
    mv x18, x1
    mv x19, x2
    mv x20, x3
    mv x21, x4

    csrr x22, 0xC00
    csrr x23, 0xC01
    csrr x24, 0xC02

    slli x5, x22, 5
    add x5, x5, x23
    bgeu x5, x21, done

    slli x6, x5, 8
    add x6, x18, x6

    slli x7, x5, 2
    add x9, x19, x0
    add x10, x20, x7

    addi x12, x0, 0
    addi x13, x0, GEMV_PACKED_COLS

col_loop:
    lw x14, 0(x6)
    lw x15, 0(x9)
    VDOT8 x12, x14, x15

    addi x6, x6, 4
    addi x9, x9, 4
    addi x13, x13, -1
    bnez x13, col_loop

    sw x12, 0(x10)

done:
    ecall
