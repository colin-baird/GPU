find_program(RISCV_CC
    NAMES riscv64-unknown-elf-gcc riscv32-unknown-elf-gcc riscv64-elf-gcc)

if(NOT RISCV_CC)
    message(STATUS "Skipping embedding_gather benchmark target: RISC-V cross-compiler not found")
    return()
endif()

set(EMBEDDING_GATHER_KERNEL_ELF ${CMAKE_CURRENT_BINARY_DIR}/embedding_gather_kernel.elf)

add_custom_command(
    OUTPUT ${EMBEDDING_GATHER_KERNEL_ELF}
    COMMAND ${RISCV_CC}
            -march=rv32im_zicsr
            -mabi=ilp32
            -nostdlib
            -nostartfiles
            -T ${CMAKE_CURRENT_SOURCE_DIR}/link.ld
            -o ${EMBEDDING_GATHER_KERNEL_ELF}
            ${CMAKE_CURRENT_SOURCE_DIR}/embedding_gather_kernel.S
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/link.ld
        ${CMAKE_CURRENT_SOURCE_DIR}/embedding_gather_kernel.S
    VERBATIM)

add_custom_target(embedding_gather_kernel DEPENDS ${EMBEDDING_GATHER_KERNEL_ELF})

add_executable(embedding_gather_bench embedding_gather_bench.cpp)
add_dependencies(embedding_gather_bench embedding_gather_kernel)
target_link_libraries(embedding_gather_bench PRIVATE gpu_sim_lib)
target_compile_definitions(embedding_gather_bench PRIVATE EMBEDDING_GATHER_KERNEL_ELF="${EMBEDDING_GATHER_KERNEL_ELF}")

add_test(NAME embedding_gather_bench COMMAND embedding_gather_bench)
