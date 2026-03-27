# Makefile for compiling riscv-tests for the GPU simulator.
#
# Usage:
#   make              # clone riscv-tests (if needed) and compile all ELFs
#   make clone        # just clone riscv-tests
#   make clean        # remove compiled ELFs
#
# Prerequisites:
#   A RISC-V cross-compiler on PATH. Tested with:
#     - riscv64-unknown-elf-gcc (multilib, from riscv-gnu-toolchain)
#     - riscv32-unknown-elf-gcc

# --- Toolchain detection ---
CC := $(shell command -v riscv64-unknown-elf-gcc 2>/dev/null || \
              command -v riscv32-unknown-elf-gcc 2>/dev/null || \
              command -v riscv64-elf-gcc 2>/dev/null)

ifeq ($(CC),)
$(error RISC-V cross-compiler not found. Install with: brew install riscv-gnu-toolchain)
endif

# --- Paths ---
RISCV_TESTS_DIR := third_party/riscv-tests
RISCV_TESTS_URL := https://github.com/riscv-software-src/riscv-tests.git
BUILD_DIR       := build

ENV_DIR         := env/gpu_sim
MACROS_DIR      := $(RISCV_TESTS_DIR)/isa/macros/scalar
LINKER_SCRIPT   := link.ld

CFLAGS := -march=rv32im -mabi=ilp32 -nostdlib -nostartfiles \
           -I $(ENV_DIR) -I $(MACROS_DIR) -T $(LINKER_SCRIPT)

# --- Test lists ---
# Tests that require features the simulator does not support.
SKIP_UI := fence_i ma_data

RV32UI_TESTS := add addi and andi auipc \
                beq bge bgeu blt bltu bne \
                jal jalr \
                lb lbu lh lhu lw lui \
                or ori \
                sb sh simple \
                sll slli slt slti sltiu sltu \
                sra srai srl srli \
                sub sw xor xori

RV32UM_TESTS := div divu mul mulh mulhsu mulhu rem remu

# Filter out skipped tests.
RV32UI_TESTS := $(filter-out $(SKIP_UI),$(RV32UI_TESTS))

# Build targets: ELF files in build/.
UI_ELFS := $(addprefix $(BUILD_DIR)/rv32ui-p-,$(RV32UI_TESTS))
UM_ELFS := $(addprefix $(BUILD_DIR)/rv32um-p-,$(RV32UM_TESTS))
ALL_ELFS := $(UI_ELFS) $(UM_ELFS)

# --- Rules ---
.PHONY: all clean clone build-elfs

# Clone first, then build ELFs in a recursive make invocation so that
# the .S source files exist when Make evaluates the pattern rules.
all: clone
	@$(MAKE) --no-print-directory build-elfs

build-elfs: $(ALL_ELFS)
	@echo ""
	@echo "Built $(words $(ALL_ELFS)) RISC-V test ELFs in $(BUILD_DIR)/"

$(BUILD_DIR)/rv32ui-p-%: $(RISCV_TESTS_DIR)/isa/rv32ui/%.S $(LINKER_SCRIPT) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $<

$(BUILD_DIR)/rv32um-p-%: $(RISCV_TESTS_DIR)/isa/rv32um/%.S $(LINKER_SCRIPT) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $<

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clone: $(RISCV_TESTS_DIR)/.cloned

$(RISCV_TESTS_DIR)/.cloned:
	@echo "Cloning riscv-tests..."
	git clone --depth 1 $(RISCV_TESTS_URL) $(RISCV_TESTS_DIR)
	@touch $@

clean:
	rm -rf $(BUILD_DIR)
