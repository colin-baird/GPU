    .option norvc

    .equ VECTOR_LEN, 64
    .equ RSQRT_TABLE_BASE, 768

    .macro TLOOKUP rd, rs1, imm
        .insn i 0x2b, 0x0, \rd, \rs1, \imm
    .endm

    .section .text
    .globl _start

_start:
    # x1 = input base
    # x2 = output base
    # x3 = number of rows / row stride in bytes
    mv x18, x1
    mv x19, x2
    mv x20, x3

    csrr x21, 0xC00
    csrr x22, 0xC01

    slli x23, x21, 5
    add x23, x23, x22          # row index = warp_id * 32 + lane_id
    bgeu x23, x20, done

    add x24, x18, x23          # input pointer for this row
    add x25, x19, x23          # output pointer for this row

    mv x26, x24
    addi x27, x0, VECTOR_LEN
    mv x10, x0                 # sumsq accumulator

sum_loop:
    lb x11, 0(x26)
    mul x11, x11, x11
    add x10, x10, x11
    add x26, x26, x20
    addi x27, x27, -1
    bnez x27, sum_loop

    srai x11, x10, 4
    addi x12, x0, 255
    bltu x11, x12, idx_ok
    mv x11, x12

idx_ok:
    TLOOKUP x13, x11, RSQRT_TABLE_BASE

    mv x26, x24
    mv x27, x25
    addi x28, x0, VECTOR_LEN

scale_loop:
    lb x14, 0(x26)
    mul x14, x14, x13
    srai x14, x14, 15
    sb x14, 0(x27)
    add x26, x26, x20
    add x27, x27, x20
    addi x28, x28, -1
    bnez x28, scale_loop

done:
    ecall
