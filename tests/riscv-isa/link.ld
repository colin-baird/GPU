/* Linker script for riscv-tests targeting the GPU simulator.
 *
 * .text at 0x0000 -- maps to instruction memory (8KB default).
 * .data at 0x2000 -- maps to data/external memory (above instruction_mem_size_bytes).
 */

OUTPUT_ARCH(riscv)
ENTRY(_start)

SECTIONS {
    . = 0x0;
    .text : {
        *(.text.init)
        *(.text)
        *(.text.*)
    }

    . = 0x2000;
    .data : {
        *(.data)
        *(.data.*)
        *(.rodata)
        *(.rodata.*)
        *(.srodata*)
    }
    .bss : {
        *(.bss)
        *(.bss.*)
        *(.sbss*)
    }

    _end = .;
}
