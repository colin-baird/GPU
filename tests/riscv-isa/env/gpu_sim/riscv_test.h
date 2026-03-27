// Custom riscv-tests environment for the GPU simulator.
//
// Replaces the official env/p/riscv_test.h. Provides only the macros that
// test_macros.h and the individual .S test files actually need, without any
// privilege-mode CSR setup (mtvec, mepc, mret) that the simulator does not
// implement.
//
// Pass/fail convention:
//   PASS: gp (x3) == 1, exit via ecall
//   FAIL: gp (x3) == (test_number << 1) | 1, exit via ecall

#ifndef _GPU_SIM_RISCV_TEST_H
#define _GPU_SIM_RISCV_TEST_H

// Target-architecture macros expected by some test files.
#define RVTEST_RV32U
#define RVTEST_RV32M

// TESTNUM is the register used by test_macros.h to track the current test
// case number. Must be gp (x3) to match the macro convention.
#define TESTNUM gp

// Mask to XLEN bits (32 for RV32).
#define MASK_XLEN(x) ((x) & ((1 << (__riscv_xlen - 1) << 1) - 1))

#define RVTEST_CODE_BEGIN       \
    .section .text.init;        \
    .globl _start;              \
_start:

#define RVTEST_CODE_END         \
    unimp

#define RVTEST_PASS             \
    fence;                      \
    li gp, 1;                   \
    ecall

#define RVTEST_FAIL             \
    fence;                      \
    sll gp, gp, 1;             \
    or  gp, gp, 1;             \
    ecall

#define RVTEST_DATA_BEGIN       \
    .section .data;

#define RVTEST_DATA_END

#endif // _GPU_SIM_RISCV_TEST_H
