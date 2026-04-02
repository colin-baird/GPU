    .option norvc

    .equ INPUT_BYTES, 128
    .equ INPUT_WORDS, (INPUT_BYTES / 4)
    .equ TILE_OUTPUTS, 32
    .equ TILE_STRIDE_BYTES, (INPUT_WORDS * TILE_OUTPUTS * 4)
    .equ OUTPUT_STRIDE_BYTES, 4
    .equ CHUNK_ITERATIONS, (INPUT_WORDS / 4)
    .equ ACTIVATION_BASE, 512

    .macro VDOT8 rd, rs1, rs2
        .insn r 0x0b, 0x0, 0x00, \rd, \rs1, \rs2
    .endm

    .macro TLOOKUP rd, rs1, imm
        .insn i 0x2b, 0x0, \rd, \rs1, \imm
    .endm

    .section .text
    .globl _start

_start:
    # x1 = input base
    # x2 = packed weight tile base
    # x3 = output base
    # x4 = output count
    mv x18, x1
    mv x19, x2
    mv x20, x3
    mv x21, x4

    csrr x22, 0xC00
    csrr x23, 0xC01
    csrr x24, 0xC02

    # output_index = warp_id * 32 + lane_id
    slli x5, x22, 5
    add x5, x5, x23
    bgeu x5, x21, done

    # input pointer is shared by the warp
    mv x13, x18

    # Each warp owns one 32-output tile.
    slli x7, x22, 12
    add x9, x19, x7
    slli x27, x23, 2
    add x15, x9, x27

    # Output pointer for this lane.
    slli x10, x5, 2
    add x10, x20, x10

    addi x12, x0, 0
    addi x16, x0, CHUNK_ITERATIONS

k_loop:
    lw x5, 0(x13)
    lw x17, 0(x15)
    VDOT8 x12, x5, x17

    lw x5, 4(x13)
    lw x17, 128(x15)
    VDOT8 x12, x5, x17

    lw x5, 8(x13)
    lw x17, 256(x15)
    VDOT8 x12, x5, x17

    lw x5, 12(x13)
    lw x17, 384(x15)
    VDOT8 x12, x5, x17

    addi x13, x13, 16
    addi x15, x15, 512
    addi x16, x16, -1
    bnez x16, k_loop

    # Quantize the accumulator into the 0..255 range used by the SiLU table.
    srai x13, x12, 5
    addi x13, x13, 128
    TLOOKUP x14, x13, ACTIVATION_BASE
    sw x14, 0(x10)

done:
    ecall
