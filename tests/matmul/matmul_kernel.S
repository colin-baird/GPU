    .option norvc

    .equ MATMUL_N, 128
    .equ MATMUL_K, 128
    .equ MATMUL_K_CHUNKS, (MATMUL_K / 4)
    .equ MATMUL_TILE_COLS, 32
    .equ TILE_COUNT, (MATMUL_N / MATMUL_TILE_COLS)
    .equ CHUNK_ITERATIONS, (MATMUL_K_CHUNKS / 4)
    .equ B_TILE_STRIDE_BYTES, (MATMUL_K_CHUNKS * MATMUL_TILE_COLS * 4)

    .macro VDOT8 rd, rs1, rs2
        .insn r 0x0b, 0x0, 0x00, \rd, \rs1, \rs2
    .endm

    .section .text
    .globl _start

_start:
    # x1 = arg0 = packed A base
    # x2 = arg1 = tiled packed B^T base
    # x3 = arg2 = output C base
    # x4 = arg3 = row count (M)
    mv x18, x1
    mv x19, x2
    mv x20, x3
    mv x21, x4

    csrr x22, 0xC00
    csrr x23, 0xC01
    csrr x24, 0xC02

    addi x25, x0, TILE_COUNT
    li x26, B_TILE_STRIDE_BYTES
    slli x27, x23, 2

row_loop:
    bgeu x22, x21, done

    slli x5, x22, 7
    add x6, x18, x5

    slli x7, x22, 9
    add x8, x20, x7

    mv x9, x19
    add x10, x8, x27
    addi x11, x0, 0

tile_loop:
    addi x12, x0, 0
    mv x13, x6
    add x15, x9, x27
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

    sw x12, 0(x10)

    addi x11, x11, 1
    add x9, x9, x26
    addi x10, x10, 128
    bltu x11, x25, tile_loop

    add x22, x22, x24
    j row_loop

done:
    ecall
