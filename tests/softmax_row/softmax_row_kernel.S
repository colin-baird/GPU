    .option norvc

    .equ ROW_COLS, 32
    .equ INPUT_ROW_BYTES, ROW_COLS
    .equ OUTPUT_ROW_BYTES, (ROW_COLS * 4)
    .equ OUTPUT_SCALE_SHIFT, 12
    .equ OUTPUT_BASE, 0x00018000

    .macro TLOOKUP rd, rs1, imm
        .insn i 0x2b, 0x0, \rd, \rs1, \imm
    .endm

    .section .text
    .globl _start

_start:
    # x1 = logits base
    # x2 = row max base
    # x3 = row sum base
    # x4 = row count
    mv x18, x1
    mv x19, x2
    mv x20, x3
    mv x21, x4

    csrr x22, 0xC00
    csrr x23, 0xC02
    csrr x24, 0xC01

row_loop:
    bgeu x22, x21, done

    slli x5, x22, 5
    add x6, x18, x5

    slli x7, x22, 2
    add x8, x19, x7
    add x9, x20, x7

    lui x10, %hi(OUTPUT_BASE)
    addi x10, x10, %lo(OUTPUT_BASE)
    slli x11, x22, 7
    add x10, x10, x11

    lw x12, 0(x8)   # row max
    lw x13, 0(x9)   # row sum

    add x14, x6, x24
    lb x15, 0(x14)  # logit for this lane
    sub x15, x12, x15

    TLOOKUP x16, x15, 0
    slli x16, x16, OUTPUT_SCALE_SHIFT
    divu x16, x16, x13

    slli x17, x24, 2
    add x17, x10, x17
    sw x16, 0(x17)

    add x22, x22, x23
    j row_loop

done:
    ecall
