    .option norvc

    .equ GEMV_COLS, 256
    .equ GEMV_PACKED_COLS, (GEMV_COLS / 4)
    .equ LANES_PER_WARP, 32
    .equ TILE_STRIDE, (GEMV_PACKED_COLS * LANES_PER_WARP * 4)
    .equ CHUNK_STRIDE, (LANES_PER_WARP * 4)

    .macro VDOT8 rd, rs1, rs2
        .insn r 0x0b, 0x0, 0x00, \rd, \rs1, \rs2
    .endm

    .section .text
    .globl _start

_start:
    # x1 = arg0 = tiled matrix base (lane-major layout)
    # x2 = arg1 = packed input vector base
    # x3 = arg2 = output vector base
    # x4 = arg3 = row count
    mv x18, x1
    mv x19, x2
    mv x20, x3
    mv x21, x4

    csrr x22, 0xC00         # warp_id
    csrr x23, 0xC01         # lane_id
    csrr x24, 0xC02         # num_warps

    # Row index = warp_id * 32 + lane_id
    slli x5, x22, 5
    add x5, x5, x23
    bgeu x5, x21, done

    # Matrix tile pointer = matrix_base + warp_id * TILE_STRIDE + lane_id * 4
    # Adjacent lanes are 4 bytes apart → all 32 fit in one 128-byte cache line.
    li x25, TILE_STRIDE
    mul x6, x22, x25
    add x6, x18, x6
    slli x7, x23, 2
    add x6, x6, x7

    # Output pointer = output_base + row_index * 4
    slli x7, x5, 2
    add x10, x20, x7

    # Vector pointer
    add x9, x19, x0

    # Accumulator and loop counter
    addi x12, x0, 0
    addi x13, x0, GEMV_PACKED_COLS

col_loop:
    lw x14, 0(x6)           # Lane-major tile load (coalesced)
    lw x15, 0(x9)           # Broadcast vector load  (coalesced)
    VDOT8 x12, x14, x15

    addi x6, x6, CHUNK_STRIDE   # Next chunk: +128 bytes
    addi x9, x9, 4              # Next vector word: +4 bytes
    addi x13, x13, -1
    bnez x13, col_loop

    sw x12, 0(x10)

done:
    ecall
