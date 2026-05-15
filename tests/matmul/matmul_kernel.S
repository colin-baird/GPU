    .option norvc

    # M, N, K are passed at runtime through kernel_args (x4, x5, x6).
    # MATMUL_M_TILE controls how many rows of A each warp folds into a single
    # tile_n iteration: a single B-chunk is reused across MATMUL_M_TILE rows
    # before being evicted, lifting arithmetic intensity by that factor over
    # the previous 1-row-per-warp scheme. M must be a multiple of
    # MATMUL_M_TILE (host-side validation).
    .ifndef MATMUL_M_TILE
    .equ MATMUL_M_TILE, 4
    .endif

    .equ MATMUL_TILE_COLS, 32                          # SIMT lane width (HW constant)
    .equ B_CHUNK_BYTES, (MATMUL_TILE_COLS * 4)         # 128 bytes per K-chunk per tile
    .equ B_INNER_STRIDE, (B_CHUNK_BYTES * 4)           # 512 bytes per k_loop iter (4 chunks)

    .macro VDOT8 rd, rs1, rs2
        .insn r 0x0b, 0x0, 0x00, \rd, \rs1, \rs2
    .endm

    .section .text
    .globl _start
_start:
    # x1 = A_base, x2 = B_base, x3 = C_base, x4 = M, x5 = N, x6 = K
    mv x18, x1                # x18 = A_base
    mv x19, x2                # x19 = B_base
    mv x20, x3                # x20 = C_base
    mv x21, x4                # x21 = M
    mv x14, x6                # x14 = K (= bytes per A row, INT8)
    slli x28, x5, 2           # x28 = N * 4 (bytes per C row, INT32)
    srli x25, x5, 5           # x25 = TILE_COUNT = N / MATMUL_TILE_COLS
    slli x26, x6, 5           # x26 = B_TILE_STRIDE_BYTES = K * MATMUL_TILE_COLS
    srli x2,  x6, 4           # x2  = CHUNK_ITERATIONS = K / 16

    csrr x22, 0xC00           # warp_id
    csrr x23, 0xC01           # lane_id
    csrr x24, 0xC02           # num_warps

    # Convert to m-block coordinates: each warp owns MATMUL_M_TILE rows per
    # outer step, striding by num_warps * MATMUL_M_TILE.
    li   x1,  MATMUL_M_TILE
    mul  x22, x22, x1         # x22 = current m_block base row index
    mul  x24, x24, x1         # x24 = m_block stride

    slli x27, x23, 2          # x27 = lane * 4 (per-lane byte offset)

m_block_loop:
    bgeu x22, x21, done

    # Stash row 0's A base for this m-block in x3. Rows 1-3 are derived from
    # it inside tile_n_loop — three adds are cheaper than burning three
    # registers as stash slots, and the freed registers (x4/x5/x6) become
    # scratch in the inner K-loop.
    mul  x3,  x22, x14
    add  x3,  x18, x3         # row 0 base

    # x10 = C output address for (row 0 of m-block, tile_n=0, this lane).
    # Stride-128 advanced per tile_n; row-stride x28 added at store time.
    mul  x10, x22, x28
    add  x10, x20, x10
    add  x10, x10, x27

    mv   x15, x19             # x15 = B current tile base (advances by x26)
    addi x11, x0,  0          # x11 = tile_n index

tile_n_loop:
    bgeu x11, x25, m_block_advance

    # Reset the four A-row pointers (rows 1..3 derived from row 0 + i*K) and
    # the per-lane B-chunk pointer.
    mv   x13, x3
    add  x7,  x13, x14
    add  x8,  x7,  x14
    add  x9,  x8,  x14
    add  x1,  x15, x27        # x1 = per-lane B chunk pointer (lane*4 inside tile)

    # Zero the four output accumulators.
    addi x12, x0, 0
    addi x29, x0, 0
    addi x30, x0, 0
    addi x31, x0, 0

    mv   x16, x2              # K-chunk countdown

k_loop:
    # ---- Chunk 0: B-load once, fold across 4 A-rows ----
    lw   x17, 0(x1)
    lw   x5,  0(x13)
    VDOT8 x12, x5, x17
    lw   x5,  0(x7)
    VDOT8 x29, x5, x17
    lw   x5,  0(x8)
    VDOT8 x30, x5, x17
    lw   x5,  0(x9)
    VDOT8 x31, x5, x17

    # ---- Chunk 1 ----
    lw   x17, 128(x1)
    lw   x5,  4(x13)
    VDOT8 x12, x5, x17
    lw   x5,  4(x7)
    VDOT8 x29, x5, x17
    lw   x5,  4(x8)
    VDOT8 x30, x5, x17
    lw   x5,  4(x9)
    VDOT8 x31, x5, x17

    # ---- Chunk 2 ----
    lw   x17, 256(x1)
    lw   x5,  8(x13)
    VDOT8 x12, x5, x17
    lw   x5,  8(x7)
    VDOT8 x29, x5, x17
    lw   x5,  8(x8)
    VDOT8 x30, x5, x17
    lw   x5,  8(x9)
    VDOT8 x31, x5, x17

    # ---- Chunk 3 ----
    lw   x17, 384(x1)
    lw   x5,  12(x13)
    VDOT8 x12, x5, x17
    lw   x5,  12(x7)
    VDOT8 x29, x5, x17
    lw   x5,  12(x8)
    VDOT8 x30, x5, x17
    lw   x5,  12(x9)
    VDOT8 x31, x5, x17

    # Advance pointers by 16 K-elements (4 chunks).
    addi x13, x13, 16
    addi x7,  x7,  16
    addi x8,  x8,  16
    addi x9,  x9,  16
    addi x1,  x1,  B_INNER_STRIDE
    addi x16, x16, -1
    bnez x16, k_loop

    # Store the four accumulators back to C[(m_block + i), (tile_n*32 + lane)].
    sw   x12, 0(x10)
    add  x16, x10, x28
    sw   x29, 0(x16)
    add  x16, x16, x28
    sw   x30, 0(x16)
    add  x16, x16, x28
    sw   x31, 0(x16)

    # Advance to next N-tile: B tile base by stride, C output by 32 INT32s.
    add  x15, x15, x26
    addi x10, x10, B_CHUNK_BYTES
    addi x11, x11, 1
    bltu x11, x25, tile_n_loop

m_block_advance:
    add  x22, x22, x24
    j    m_block_loop

done:
    ecall
