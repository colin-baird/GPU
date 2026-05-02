find_program(RISCV_CC
    NAMES riscv64-unknown-elf-gcc riscv32-unknown-elf-gcc riscv64-elf-gcc)

if(NOT RISCV_CC)
    message(STATUS "Skipping panic_drain_test benchmark target: RISC-V cross-compiler not found")
    return()
endif()

set(PANIC_DRAIN_TEST_KERNEL_ELF ${CMAKE_CURRENT_BINARY_DIR}/panic_drain_test_kernel.elf)

add_custom_command(
    OUTPUT ${PANIC_DRAIN_TEST_KERNEL_ELF}
    COMMAND ${RISCV_CC}
            -march=rv32im_zicsr
            -mabi=ilp32
            -nostdlib
            -nostartfiles
            -T ${CMAKE_CURRENT_SOURCE_DIR}/link.ld
            -o ${PANIC_DRAIN_TEST_KERNEL_ELF}
            ${CMAKE_CURRENT_SOURCE_DIR}/panic_drain_test_kernel.S
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/link.ld
        ${CMAKE_CURRENT_SOURCE_DIR}/panic_drain_test_kernel.S
    VERBATIM)

add_custom_target(panic_drain_test_kernel DEPENDS ${PANIC_DRAIN_TEST_KERNEL_ELF})

add_executable(panic_drain_test_bench panic_drain_test_bench.cpp)
add_dependencies(panic_drain_test_bench panic_drain_test_kernel)
target_link_libraries(panic_drain_test_bench PRIVATE gpu_sim_lib)
target_compile_definitions(panic_drain_test_bench PRIVATE PANIC_DRAIN_TEST_KERNEL_ELF="${PANIC_DRAIN_TEST_KERNEL_ELF}")

add_test(NAME panic_drain_test_bench COMMAND panic_drain_test_bench)
