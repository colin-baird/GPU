#!/usr/bin/env python3
"""
Analytical performance reference generator for RISC-V ISA tests.

Disassembles an ELF, traces the dynamic execution path, classifies
instructions, and applies the architectural spec's timing rules to
produce reference performance statistics.

All timing derivations are from the architectural spec
(resources/gpu_architectural_spec.md), NOT from the timing model code.

Usage:
    python3 analyze_elf.py <elf_path> [--output <json_path>]
    python3 analyze_elf.py --batch <elf_dir> --output-dir <json_dir>
"""

import argparse
import copy
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration (default SimConfig values)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "num_warps": 1,
    "instruction_buffer_depth": 2,
    "multiply_pipeline_stages": 3,
    "num_ldst_units": 8,
    "l1_cache_size_bytes": 4096,
    "cache_line_size_bytes": 128,
    "num_mshrs": 4,
    "write_buffer_depth": 4,
    "external_memory_latency_cycles": 100,
}


# ---------------------------------------------------------------------------
# Instruction classification
# ---------------------------------------------------------------------------
class ExecUnit(Enum):
    ALU = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    LDST = auto()
    TLOOKUP = auto()
    SYSTEM = auto()


class InstrKind(Enum):
    ALU = auto()        # R-type ALU, I-type ALU, LUI, AUIPC
    BRANCH = auto()     # BEQ, BNE, BLT, BGE, BLTU, BGEU
    JAL = auto()        # JAL
    JALR = auto()       # JALR
    LOAD = auto()       # LB, LBU, LH, LHU, LW
    STORE = auto()      # SB, SH, SW
    MUL = auto()        # MUL, MULH, MULHSU, MULHU
    DIV = auto()        # DIV, DIVU, REM, REMU
    FENCE = auto()
    ECALL = auto()
    EBREAK = auto()
    CSR = auto()
    NOP = auto()        # Pseudo: nop (addi x0, x0, 0)
    UNKNOWN = auto()


@dataclass
class Instruction:
    addr: int
    raw_hex: str
    mnemonic: str
    operands: str
    kind: InstrKind = InstrKind.UNKNOWN
    unit: ExecUnit = ExecUnit.ALU
    rd: Optional[int] = None
    rs1: Optional[int] = None
    rs2: Optional[int] = None
    has_rd: bool = False
    is_jump: bool = False       # JAL/JALR - always taken
    branch_target: Optional[int] = None
    mem_addr: Optional[int] = None  # Effective address for loads/stores


# Register name -> number mapping
REG_MAP = {f"x{i}": i for i in range(32)}
REG_MAP.update({
    "zero": 0, "ra": 1, "sp": 2, "gp": 3, "tp": 4,
    "t0": 5, "t1": 6, "t2": 7,
    "s0": 8, "fp": 8, "s1": 9,
    "a0": 10, "a1": 11, "a2": 12, "a3": 13, "a4": 14, "a5": 15, "a6": 16, "a7": 17,
    "s2": 18, "s3": 19, "s4": 20, "s5": 21, "s6": 22, "s7": 23,
    "s8": 24, "s9": 25, "s10": 26, "s11": 27,
    "t3": 28, "t4": 29, "t5": 30, "t6": 31,
})


def parse_reg(s: str) -> Optional[int]:
    """Parse a register name like 'a1' or 'x11' to a register number."""
    s = s.strip().rstrip(",")
    return REG_MAP.get(s)


# Mnemonic -> (kind, unit, has_rd)
MNEMONIC_TABLE = {
    # ALU R-type
    "add": (InstrKind.ALU, ExecUnit.ALU, True),
    "sub": (InstrKind.ALU, ExecUnit.ALU, True),
    "xor": (InstrKind.ALU, ExecUnit.ALU, True),
    "or":  (InstrKind.ALU, ExecUnit.ALU, True),
    "and": (InstrKind.ALU, ExecUnit.ALU, True),
    "sll": (InstrKind.ALU, ExecUnit.ALU, True),
    "srl": (InstrKind.ALU, ExecUnit.ALU, True),
    "sra": (InstrKind.ALU, ExecUnit.ALU, True),
    "slt": (InstrKind.ALU, ExecUnit.ALU, True),
    "sltu": (InstrKind.ALU, ExecUnit.ALU, True),
    # ALU I-type
    "addi": (InstrKind.ALU, ExecUnit.ALU, True),
    "xori": (InstrKind.ALU, ExecUnit.ALU, True),
    "ori":  (InstrKind.ALU, ExecUnit.ALU, True),
    "andi": (InstrKind.ALU, ExecUnit.ALU, True),
    "slli": (InstrKind.ALU, ExecUnit.ALU, True),
    "srli": (InstrKind.ALU, ExecUnit.ALU, True),
    "srai": (InstrKind.ALU, ExecUnit.ALU, True),
    "slti": (InstrKind.ALU, ExecUnit.ALU, True),
    "sltiu": (InstrKind.ALU, ExecUnit.ALU, True),
    # U-type
    "lui":   (InstrKind.ALU, ExecUnit.ALU, True),
    "auipc": (InstrKind.ALU, ExecUnit.ALU, True),
    # Pseudo
    "li":    (InstrKind.ALU, ExecUnit.ALU, True),  # expands to addi or lui+addi
    "mv":    (InstrKind.ALU, ExecUnit.ALU, True),  # expands to addi rd, rs, 0
    "nop":   (InstrKind.NOP, ExecUnit.ALU, False),  # addi x0, x0, 0
    "not":   (InstrKind.ALU, ExecUnit.ALU, True),
    "neg":   (InstrKind.ALU, ExecUnit.ALU, True),
    "seqz":  (InstrKind.ALU, ExecUnit.ALU, True),
    "snez":  (InstrKind.ALU, ExecUnit.ALU, True),
    "sltz":  (InstrKind.ALU, ExecUnit.ALU, True),
    "sgtz":  (InstrKind.ALU, ExecUnit.ALU, True),
    # Branches
    "beq":   (InstrKind.BRANCH, ExecUnit.ALU, False),
    "bne":   (InstrKind.BRANCH, ExecUnit.ALU, False),
    "blt":   (InstrKind.BRANCH, ExecUnit.ALU, False),
    "bge":   (InstrKind.BRANCH, ExecUnit.ALU, False),
    "bltu":  (InstrKind.BRANCH, ExecUnit.ALU, False),
    "bgeu":  (InstrKind.BRANCH, ExecUnit.ALU, False),
    # Pseudo branches
    "beqz":  (InstrKind.BRANCH, ExecUnit.ALU, False),
    "bnez":  (InstrKind.BRANCH, ExecUnit.ALU, False),
    "blez":  (InstrKind.BRANCH, ExecUnit.ALU, False),
    "bgez":  (InstrKind.BRANCH, ExecUnit.ALU, False),
    "bltz":  (InstrKind.BRANCH, ExecUnit.ALU, False),
    "bgtz":  (InstrKind.BRANCH, ExecUnit.ALU, False),
    # Jumps
    "jal":   (InstrKind.JAL, ExecUnit.ALU, True),
    "jalr":  (InstrKind.JALR, ExecUnit.ALU, True),
    "j":     (InstrKind.JAL, ExecUnit.ALU, False),   # jal x0, offset
    "jr":    (InstrKind.JALR, ExecUnit.ALU, False),   # jalr x0, rs, 0
    "ret":   (InstrKind.JALR, ExecUnit.ALU, False),   # jalr x0, ra, 0
    "call":  (InstrKind.JAL, ExecUnit.ALU, True),     # pseudo: auipc+jalr
    "tail":  (InstrKind.JAL, ExecUnit.ALU, False),    # pseudo: auipc+jalr
    # Loads
    "lb":   (InstrKind.LOAD, ExecUnit.LDST, True),
    "lbu":  (InstrKind.LOAD, ExecUnit.LDST, True),
    "lh":   (InstrKind.LOAD, ExecUnit.LDST, True),
    "lhu":  (InstrKind.LOAD, ExecUnit.LDST, True),
    "lw":   (InstrKind.LOAD, ExecUnit.LDST, True),
    "la":   (InstrKind.ALU, ExecUnit.ALU, True),  # pseudo: auipc+addi
    # Stores
    "sb":   (InstrKind.STORE, ExecUnit.LDST, False),
    "sh":   (InstrKind.STORE, ExecUnit.LDST, False),
    "sw":   (InstrKind.STORE, ExecUnit.LDST, False),
    # Multiply
    "mul":    (InstrKind.MUL, ExecUnit.MULTIPLY, True),
    "mulh":   (InstrKind.MUL, ExecUnit.MULTIPLY, True),
    "mulhsu": (InstrKind.MUL, ExecUnit.MULTIPLY, True),
    "mulhu":  (InstrKind.MUL, ExecUnit.MULTIPLY, True),
    # Divide
    "div":   (InstrKind.DIV, ExecUnit.DIVIDE, True),
    "divu":  (InstrKind.DIV, ExecUnit.DIVIDE, True),
    "rem":   (InstrKind.DIV, ExecUnit.DIVIDE, True),
    "remu":  (InstrKind.DIV, ExecUnit.DIVIDE, True),
    # System
    "fence":  (InstrKind.FENCE, ExecUnit.SYSTEM, False),
    "ecall":  (InstrKind.ECALL, ExecUnit.SYSTEM, False),
    "ebreak": (InstrKind.EBREAK, ExecUnit.SYSTEM, False),
    "unimp":  (InstrKind.UNKNOWN, ExecUnit.SYSTEM, False),
    # CSR
    "csrr":   (InstrKind.CSR, ExecUnit.ALU, True),
    "csrrs":  (InstrKind.CSR, ExecUnit.ALU, True),
    "csrrw":  (InstrKind.CSR, ExecUnit.ALU, True),
    "csrrc":  (InstrKind.CSR, ExecUnit.ALU, True),
    "csrrsi": (InstrKind.CSR, ExecUnit.ALU, True),
    "csrrwi": (InstrKind.CSR, ExecUnit.ALU, True),
    "csrrci": (InstrKind.CSR, ExecUnit.ALU, True),
}


def find_objdump():
    """Find RISC-V objdump on system."""
    for name in ["riscv64-unknown-elf-objdump", "riscv32-unknown-elf-objdump",
                  "riscv64-elf-objdump"]:
        try:
            subprocess.run([name, "--version"], capture_output=True, check=True)
            return name
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    raise RuntimeError("No RISC-V objdump found")


def disassemble(elf_path: str) -> list[Instruction]:
    """Disassemble an ELF and return a list of instructions."""
    objdump = find_objdump()
    result = subprocess.run(
        [objdump, "-d", elf_path],
        capture_output=True, text=True, check=True
    )

    instructions = []
    # Match lines like: "   0:	0ff0000f          	fence"
    pattern = re.compile(r"^\s*([0-9a-f]+):\s+([0-9a-f]+)\s+(\S+)\s*(.*?)\s*$")

    for line in result.stdout.splitlines():
        m = pattern.match(line)
        if not m:
            continue

        addr = int(m.group(1), 16)
        raw_hex = m.group(2)
        mnemonic = m.group(3)
        operands = m.group(4)

        instr = Instruction(
            addr=addr,
            raw_hex=raw_hex,
            mnemonic=mnemonic,
            operands=operands,
        )

        # Classify
        lookup = MNEMONIC_TABLE.get(mnemonic)
        if lookup:
            instr.kind, instr.unit, instr.has_rd = lookup
        else:
            instr.kind = InstrKind.UNKNOWN

        # Parse registers and branch targets
        _parse_operands(instr)

        instructions.append(instr)

    return instructions


def _parse_operands(instr: Instruction):
    """Parse operand string to extract registers and branch targets."""
    ops = instr.operands
    if not ops:
        return

    # Extract branch/jump target from operand string
    # Format: "a4,t2,4e0 <fail>" or "4f0 <pass>" or "offset(reg)"
    target_match = re.search(r'\b([0-9a-f]+)\s+<', ops)

    if instr.kind == InstrKind.BRANCH:
        parts = [p.strip() for p in ops.split(",")]
        if len(parts) >= 2:
            instr.rs1 = parse_reg(parts[0])
            instr.rs2 = parse_reg(parts[1])
        elif len(parts) == 1:
            # Pseudo branch like beqz rs, target
            instr.rs1 = parse_reg(parts[0])
            instr.rs2 = 0  # implicit x0
        if target_match:
            instr.branch_target = int(target_match.group(1), 16)

    elif instr.kind in (InstrKind.JAL, InstrKind.JALR):
        instr.is_jump = True
        parts = [p.strip() for p in ops.split(",")]
        if instr.mnemonic in ("j", "jr", "ret"):
            instr.has_rd = False
            if instr.mnemonic == "j" and target_match:
                instr.branch_target = int(target_match.group(1), 16)
            elif instr.mnemonic in ("jr", "ret") and parts:
                instr.rs1 = parse_reg(parts[0])
        elif instr.mnemonic == "jal":
            if len(parts) >= 1:
                instr.rd = parse_reg(parts[0])
                if instr.rd is not None and instr.rd != 0:
                    instr.has_rd = True
                else:
                    instr.has_rd = False
            if target_match:
                instr.branch_target = int(target_match.group(1), 16)
        elif instr.mnemonic == "jalr":
            if len(parts) >= 1:
                instr.rd = parse_reg(parts[0])
                if instr.rd is not None and instr.rd != 0:
                    instr.has_rd = True
                else:
                    instr.has_rd = False
            if len(parts) >= 2:
                # Parse "offset(rs1)" or just "rs1"
                base_match = re.search(r'([a-z0-9]+)\)', parts[1])
                if base_match:
                    instr.rs1 = parse_reg(base_match.group(1))
                else:
                    instr.rs1 = parse_reg(parts[1])

    elif instr.kind in (InstrKind.ALU, InstrKind.MUL, InstrKind.DIV, InstrKind.CSR):
        parts = [p.strip() for p in ops.split(",")]
        if len(parts) >= 1:
            instr.rd = parse_reg(parts[0])
            if instr.rd == 0:
                instr.has_rd = False
        if len(parts) >= 2:
            instr.rs1 = parse_reg(parts[1])
        if len(parts) >= 3:
            instr.rs2 = parse_reg(parts[2])

    elif instr.kind == InstrKind.NOP:
        instr.has_rd = False
        instr.rd = 0

    elif instr.kind == InstrKind.LOAD:
        parts = [p.strip() for p in ops.split(",")]
        if len(parts) >= 1:
            instr.rd = parse_reg(parts[0])
            if instr.rd == 0:
                instr.has_rd = False
        if len(parts) >= 2:
            base_match = re.search(r'\((\w+)\)', parts[1])
            if base_match:
                instr.rs1 = parse_reg(base_match.group(1))

    elif instr.kind == InstrKind.STORE:
        parts = [p.strip() for p in ops.split(",")]
        if len(parts) >= 1:
            instr.rs2 = parse_reg(parts[0])  # store data
        if len(parts) >= 2:
            base_match = re.search(r'\((\w+)\)', parts[1])
            if base_match:
                instr.rs1 = parse_reg(base_match.group(1))


# ---------------------------------------------------------------------------
# Minimal RV32IM interpreter for execution path tracing
# ---------------------------------------------------------------------------
MASK32 = 0xFFFFFFFF


def _sign_extend(val: int, bits: int) -> int:
    """Sign-extend a value from `bits` width to 32-bit Python int."""
    if val & (1 << (bits - 1)):
        val -= (1 << bits)
    return val


def _to_signed32(val: int) -> int:
    """Convert unsigned 32-bit to signed."""
    val &= MASK32
    if val & 0x80000000:
        return val - 0x100000000
    return val


def _to_unsigned32(val: int) -> int:
    return val & MASK32


def _load_data_section(elf_path: str) -> dict:
    """Load the .data section from an ELF using objdump -s."""
    objdump = find_objdump()
    try:
        result = subprocess.run(
            [objdump, "-s", "-j", ".data", elf_path],
            capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError:
        return {}

    memory = {}
    # Parse lines like: " 2000 ff00ff00 00ff00ff f00ff00f 0ff00ff0  ..."
    for line in result.stdout.splitlines():
        m = re.match(r'^\s+([0-9a-f]+)\s+((?:[0-9a-f]+\s*)+)', line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        hex_data = m.group(2).strip().replace(" ", "")
        for i in range(0, len(hex_data), 2):
            if i + 2 <= len(hex_data):
                memory[addr + i // 2] = int(hex_data[i:i+2], 16)

    return memory


def _mem_read(memory: dict, addr: int, size: int) -> int:
    """Read size bytes from memory (little-endian)."""
    val = 0
    for i in range(size):
        val |= memory.get((addr + i) & MASK32, 0) << (i * 8)
    return val


def _mem_write(memory: dict, addr: int, size: int, val: int):
    """Write size bytes to memory (little-endian)."""
    for i in range(size):
        memory[(addr + i) & MASK32] = (val >> (i * 8)) & 0xFF


def trace_isa_test(instructions: list[Instruction],
                   elf_path: str = "") -> list[Instruction]:
    """Trace the dynamic execution path using a minimal RV32IM interpreter.

    Simulates register values to correctly resolve all branch conditions,
    including complex patterns in branch tests (beq, bne, etc.).
    """
    # Build address -> instruction map and raw instruction map
    addr_map = {i.addr: i for i in instructions}
    raw_map = {i.addr: int(i.raw_hex, 16) for i in instructions}

    # Load data section for load/store tests
    memory = _load_data_section(elf_path) if elf_path else {}

    regs = [0] * 32  # x0..x31, x0 always 0
    pc = 0
    trace = []
    max_steps = 100000

    for _ in range(max_steps):
        if pc not in addr_map:
            break

        instr = copy.copy(addr_map[pc])
        raw = raw_map[pc]
        trace.append(instr)

        if instr.kind == InstrKind.ECALL or instr.kind == InstrKind.EBREAK:
            break

        next_pc = pc + 4

        # Decode and execute from raw instruction word
        opcode = raw & 0x7F
        rd_idx = (raw >> 7) & 0x1F
        funct3 = (raw >> 12) & 0x7
        rs1_idx = (raw >> 15) & 0x1F
        rs2_idx = (raw >> 20) & 0x1F
        funct7 = (raw >> 25) & 0x7F

        rs1_val = _to_signed32(regs[rs1_idx])
        rs2_val = _to_signed32(regs[rs2_idx])
        rs1_u = _to_unsigned32(regs[rs1_idx])
        rs2_u = _to_unsigned32(regs[rs2_idx])

        # I-type immediate
        imm_i = _sign_extend((raw >> 20) & 0xFFF, 12)
        # S-type immediate
        imm_s = _sign_extend(((raw >> 25) << 5) | ((raw >> 7) & 0x1F), 12)
        # B-type immediate
        imm_b = _sign_extend(
            (((raw >> 31) & 1) << 12) | (((raw >> 7) & 1) << 11) |
            (((raw >> 25) & 0x3F) << 5) | (((raw >> 8) & 0xF) << 1), 13)
        # U-type immediate
        imm_u = raw & 0xFFFFF000
        # J-type immediate
        imm_j = _sign_extend(
            (((raw >> 31) & 1) << 20) | (((raw >> 12) & 0xFF) << 12) |
            (((raw >> 20) & 1) << 11) | (((raw >> 21) & 0x3FF) << 1), 21)

        result = 0

        if opcode == 0b0110011:  # R-type ALU / M-ext
            if funct7 == 0x01:  # M-extension
                if funct3 == 0:  # MUL
                    result = (rs1_val * rs2_val) & MASK32
                elif funct3 == 1:  # MULH
                    result = ((rs1_val * rs2_val) >> 32) & MASK32
                elif funct3 == 2:  # MULHSU
                    result = ((rs1_val * rs2_u) >> 32) & MASK32
                elif funct3 == 3:  # MULHU
                    result = ((rs1_u * rs2_u) >> 32) & MASK32
                elif funct3 == 4:  # DIV
                    if rs2_val == 0:
                        result = MASK32
                    elif rs1_val == -0x80000000 and rs2_val == -1:
                        result = 0x80000000
                    else:
                        # Python division truncates toward negative infinity;
                        # RISC-V truncates toward zero
                        q = abs(rs1_val) // abs(rs2_val)
                        if (rs1_val < 0) != (rs2_val < 0):
                            q = -q
                        result = _to_unsigned32(q)
                elif funct3 == 5:  # DIVU
                    result = (rs1_u // rs2_u if rs2_u != 0 else MASK32) & MASK32
                elif funct3 == 6:  # REM
                    if rs2_val == 0:
                        result = _to_unsigned32(rs1_val)
                    elif rs1_val == -0x80000000 and rs2_val == -1:
                        result = 0
                    else:
                        r = abs(rs1_val) % abs(rs2_val)
                        if rs1_val < 0:
                            r = -r
                        result = _to_unsigned32(r)
                elif funct3 == 7:  # REMU
                    result = (rs1_u % rs2_u if rs2_u != 0 else rs1_u) & MASK32
            else:
                if funct3 == 0:
                    result = (rs1_val - rs2_val if funct7 == 0x20 else rs1_val + rs2_val) & MASK32
                elif funct3 == 1:
                    result = (rs1_u << (rs2_u & 0x1F)) & MASK32
                elif funct3 == 2:
                    result = 1 if rs1_val < rs2_val else 0
                elif funct3 == 3:
                    result = 1 if rs1_u < rs2_u else 0
                elif funct3 == 4:
                    result = (rs1_u ^ rs2_u) & MASK32
                elif funct3 == 5:
                    shamt = rs2_u & 0x1F
                    result = (rs1_u >> shamt if funct7 != 0x20
                              else _to_unsigned32(rs1_val >> shamt))
                elif funct3 == 6:
                    result = (rs1_u | rs2_u) & MASK32
                elif funct3 == 7:
                    result = (rs1_u & rs2_u) & MASK32

            if rd_idx != 0:
                regs[rd_idx] = result & MASK32

        elif opcode == 0b0010011:  # I-type ALU
            if funct3 == 0:
                result = (rs1_val + imm_i) & MASK32
            elif funct3 == 1:
                result = (rs1_u << (imm_i & 0x1F)) & MASK32
            elif funct3 == 2:
                result = 1 if rs1_val < imm_i else 0
            elif funct3 == 3:
                result = 1 if rs1_u < (_to_unsigned32(imm_i)) else 0
            elif funct3 == 4:
                result = (rs1_u ^ _to_unsigned32(imm_i)) & MASK32
            elif funct3 == 5:
                shamt = imm_i & 0x1F
                if funct7 == 0x20:
                    result = _to_unsigned32(rs1_val >> shamt)
                else:
                    result = (rs1_u >> shamt) & MASK32
            elif funct3 == 6:
                result = (rs1_u | _to_unsigned32(imm_i)) & MASK32
            elif funct3 == 7:
                result = (rs1_u & _to_unsigned32(imm_i)) & MASK32

            if rd_idx != 0:
                regs[rd_idx] = result & MASK32

        elif opcode == 0b0110111:  # LUI
            if rd_idx != 0:
                regs[rd_idx] = imm_u & MASK32

        elif opcode == 0b0010111:  # AUIPC
            if rd_idx != 0:
                regs[rd_idx] = (pc + _to_signed32(imm_u)) & MASK32

        elif opcode == 0b1101111:  # JAL
            if rd_idx != 0:
                regs[rd_idx] = (pc + 4) & MASK32
            next_pc = (pc + imm_j) & MASK32

        elif opcode == 0b1100111:  # JALR
            target = (rs1_val + imm_i) & ~1 & MASK32
            if rd_idx != 0:
                regs[rd_idx] = (pc + 4) & MASK32
            next_pc = target

        elif opcode == 0b1100011:  # BRANCH
            taken = False
            if funct3 == 0:    # BEQ
                taken = rs1_val == rs2_val
            elif funct3 == 1:  # BNE
                taken = rs1_val != rs2_val
            elif funct3 == 4:  # BLT
                taken = rs1_val < rs2_val
            elif funct3 == 5:  # BGE
                taken = rs1_val >= rs2_val
            elif funct3 == 6:  # BLTU
                taken = rs1_u < rs2_u
            elif funct3 == 7:  # BGEU
                taken = rs1_u >= rs2_u
            if taken:
                next_pc = (pc + imm_b) & MASK32

        elif opcode == 0b0000011:  # LOAD
            addr = (rs1_val + imm_i) & MASK32
            instr.mem_addr = addr
            if funct3 == 0:     # LB
                val = _mem_read(memory, addr, 1)
                result = _sign_extend(val, 8) & MASK32
            elif funct3 == 1:   # LH
                val = _mem_read(memory, addr, 2)
                result = _sign_extend(val, 16) & MASK32
            elif funct3 == 2:   # LW
                result = _mem_read(memory, addr, 4) & MASK32
            elif funct3 == 4:   # LBU
                result = _mem_read(memory, addr, 1)
            elif funct3 == 5:   # LHU
                result = _mem_read(memory, addr, 2)
            else:
                result = 0
            if rd_idx != 0:
                regs[rd_idx] = result & MASK32

        elif opcode == 0b0100011:  # STORE
            addr = (rs1_val + imm_s) & MASK32
            instr.mem_addr = addr
            if funct3 == 0:     # SB
                _mem_write(memory, addr, 1, rs2_u)
            elif funct3 == 1:   # SH
                _mem_write(memory, addr, 2, rs2_u)
            elif funct3 == 2:   # SW
                _mem_write(memory, addr, 4, rs2_u)

        elif opcode == 0b0001111:  # FENCE
            pass  # NOP

        # Keep x0 always 0
        regs[0] = 0
        pc = next_pc

    return trace


# ---------------------------------------------------------------------------
# Pipeline timing model (analytical, from spec)
# ---------------------------------------------------------------------------
@dataclass
class PipelineState:
    """Track pipeline state for cycle counting."""
    cycle: int = 0
    scoreboard: dict = field(default_factory=dict)  # reg -> clear_cycle

    # Stats accumulators
    total_instructions: int = 0
    branch_flushes: int = 0
    alu_instructions: int = 0
    mul_instructions: int = 0
    div_instructions: int = 0
    ldst_instructions: int = 0
    fence_instructions: int = 0
    system_instructions: int = 0
    nop_instructions: int = 0

    # Memory tracking
    cache_lines_loaded: set = field(default_factory=set)  # set of line addresses
    load_hits: int = 0
    load_misses: int = 0
    store_hits: int = 0
    store_misses: int = 0
    coalesced_requests: int = 0

    # Stall tracking
    stall_scoreboard: int = 0
    stall_unit_busy: int = 0

    # Unit busy tracking
    div_busy_until: int = 0
    mul_busy_until: int = 0


def compute_timing(trace: list[Instruction], config: dict) -> dict:
    """Compute expected timing statistics for a traced instruction sequence.

    Applies the architectural spec's timing rules to derive cycle-accurate
    expected statistics.
    """
    mul_stages = config["multiply_pipeline_stages"]
    num_ldst = config["num_ldst_units"]
    cache_line_size = config["cache_line_size_bytes"]
    cache_size = config["l1_cache_size_bytes"]
    mem_latency = config["external_memory_latency_cycles"]
    num_sets = cache_size // cache_line_size

    state = PipelineState()

    # Pipeline fill: 2 cycles before first issue
    state.cycle = 2
    # Track the cycle at which the next issue can happen
    next_issue_cycle = 3  # First issue at cycle 3

    for i, instr in enumerate(trace):
        # Determine earliest issue cycle considering:
        # 1. Pipeline fill / previous issue
        # 2. Scoreboard stalls (RAW dependencies)
        # 3. Unit busy (DIV is non-pipelined)
        # 4. Operand collector busy (always 1 cycle for 2-operand)

        issue_cycle = next_issue_cycle

        # Check scoreboard for source register dependencies
        src_regs = []
        if instr.rs1 is not None and instr.rs1 != 0:
            src_regs.append(instr.rs1)
        if instr.rs2 is not None and instr.rs2 != 0:
            src_regs.append(instr.rs2)

        for reg in src_regs:
            if reg in state.scoreboard:
                clear_cycle = state.scoreboard[reg]
                if clear_cycle >= issue_cycle:
                    stall = clear_cycle - issue_cycle + 1
                    state.stall_scoreboard += stall
                    issue_cycle = clear_cycle + 1

        # Check unit busy (DIV is non-pipelined)
        if instr.kind == InstrKind.DIV and state.div_busy_until >= issue_cycle:
            stall = state.div_busy_until - issue_cycle + 1
            state.stall_unit_busy += stall
            issue_cycle = state.div_busy_until + 1

        state.cycle = issue_cycle
        state.total_instructions += 1

        # Compute scoreboard clear cycle for destination register
        # This is the cycle at which the scoreboard is committed (end of cycle)
        if instr.has_rd and instr.rd is not None and instr.rd != 0:
            if instr.kind == InstrKind.ALU or instr.kind == InstrKind.NOP:
                # ALU: clears same cycle as issue (spec SS 4.6: 1 cycle)
                clear = issue_cycle
            elif instr.kind in (InstrKind.JAL, InstrKind.JALR):
                # JAL/JALR: ALU path, clears same cycle
                clear = issue_cycle
            elif instr.kind == InstrKind.CSR:
                # CSR: routed through ALU, 1-cycle
                clear = issue_cycle
            elif instr.kind == InstrKind.MUL:
                # MUL: pipelined, mul_stages cycles after dispatch
                # dispatch same cycle as issue (1-cycle opcoll)
                # stages count down: stages -> 0 over (stages-1) cycles
                clear = issue_cycle + mul_stages - 1
            elif instr.kind == InstrKind.DIV:
                # DIV: 32 cycles iterative
                clear = issue_cycle + 31
                state.div_busy_until = issue_cycle + 31
            elif instr.kind == InstrKind.LOAD:
                # LOAD: addr gen + coalescing + cache
                addr_gen_cycles = -(-32 // num_ldst)  # ceil(32/num_ldst)
                # In ISA tests, all threads same address -> coalesced
                # Use actual effective address from trace
                mem_addr = instr.mem_addr if instr.mem_addr is not None else 0x2000
                line_addr = mem_addr // cache_line_size
                tag = line_addr // num_sets
                set_index = line_addr % num_sets

                # Direct-mapped: check if same tag is in this set
                cache_key = (set_index, tag)
                if cache_key in state.cache_lines_loaded:
                    # Cache hit
                    state.load_hits += 1
                    clear = issue_cycle + addr_gen_cycles - 1
                else:
                    # Cache miss - install new line (may evict)
                    # Remove any existing entry for this set (direct-mapped)
                    state.cache_lines_loaded = {
                        k for k in state.cache_lines_loaded if k[0] != set_index
                    }
                    state.cache_lines_loaded.add(cache_key)
                    state.load_misses += 1
                    clear = issue_cycle + addr_gen_cycles - 1 + mem_latency
                state.coalesced_requests += 1
            else:
                clear = issue_cycle

            state.scoreboard[instr.rd] = clear

        # Count instruction types
        if instr.kind == InstrKind.ALU:
            state.alu_instructions += 1
        elif instr.kind == InstrKind.NOP:
            state.alu_instructions += 1  # NOP is ALU (addi x0, x0, 0)
            state.nop_instructions += 1
        elif instr.kind in (InstrKind.BRANCH, InstrKind.JAL, InstrKind.JALR):
            state.alu_instructions += 1  # All go through ALU unit
        elif instr.kind == InstrKind.MUL:
            state.mul_instructions += 1
        elif instr.kind == InstrKind.DIV:
            state.div_instructions += 1
        elif instr.kind in (InstrKind.LOAD, InstrKind.STORE):
            state.ldst_instructions += 1
        elif instr.kind == InstrKind.FENCE:
            state.fence_instructions += 1
        elif instr.kind in (InstrKind.ECALL, InstrKind.EBREAK):
            state.system_instructions += 1
        elif instr.kind == InstrKind.CSR:
            state.alu_instructions += 1  # CSR routed through ALU

        # Handle stores (cache tracking, write-allocate)
        if instr.kind == InstrKind.STORE:
            mem_addr = instr.mem_addr if instr.mem_addr is not None else 0x2000
            line_addr = mem_addr // cache_line_size
            tag = line_addr // num_sets
            set_index = line_addr % num_sets
            cache_key = (set_index, tag)
            if cache_key in state.cache_lines_loaded:
                state.store_hits += 1
            else:
                # Write-allocate: fetch line then write
                state.cache_lines_loaded = {
                    k for k in state.cache_lines_loaded if k[0] != set_index
                }
                state.cache_lines_loaded.add(cache_key)
                state.store_misses += 1
            state.coalesced_requests += 1

        # Branch penalty: taken branches add 2 idle cycles
        is_taken_branch = False
        if instr.kind == InstrKind.BRANCH:
            # Check if this branch was taken by looking at the next instruction
            if i + 1 < len(trace):
                next_addr = trace[i + 1].addr
                if next_addr != instr.addr + 4:
                    is_taken_branch = True
        elif instr.kind in (InstrKind.JAL, InstrKind.JALR):
            # JAL/JALR always redirect PC
            is_taken_branch = True

        if is_taken_branch:
            state.branch_flushes += 1
            # 2-cycle penalty: next issue at issue_cycle + 3
            next_issue_cycle = issue_cycle + 3
        else:
            # Normal: next issue at issue_cycle + 1
            next_issue_cycle = issue_cycle + 1

    # After last instruction (ECALL), the pipeline needs to drain.
    # ECALL dispatches to SYSTEM (no execution unit), warp goes inactive.
    # all_warps_done check happens at end of the ECALL's cycle.
    # Total cycles = the cycle of the last instruction issued.
    total_cycles = state.cycle

    # Build the reference stats
    cache_hits = state.load_hits + state.store_hits
    cache_misses = state.load_misses + state.store_misses
    external_reads = state.load_misses
    external_writes = state.store_hits + state.store_misses  # write-through

    # Scheduler idle cycles = pipeline_fill (2) + branch penalty gaps
    scheduler_idle = total_cycles - state.total_instructions

    return {
        "total_cycles": total_cycles,
        "total_instructions_issued": state.total_instructions,
        "branch_flushes": state.branch_flushes,
        "scheduler_idle_cycles": scheduler_idle,
        "alu_instructions": state.alu_instructions,
        "mul_instructions": state.mul_instructions,
        "div_instructions": state.div_instructions,
        "ldst_instructions": state.ldst_instructions,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "load_hits": state.load_hits,
        "load_misses": state.load_misses,
        "store_hits": state.store_hits,
        "store_misses": state.store_misses,
        "coalesced_requests": state.coalesced_requests,
        "serialized_requests": 0,
        "external_memory_reads": external_reads,
        "external_memory_writes": external_writes,
        "warp_instructions": [state.total_instructions],
        "warp_stall_scoreboard": state.stall_scoreboard,
        "warp_stall_unit_busy": state.stall_unit_busy,
    }


# ---------------------------------------------------------------------------
# Reference JSON generation
# ---------------------------------------------------------------------------
def generate_reference(elf_path: str, config: dict) -> dict:
    """Generate a complete reference JSON for an ELF."""
    test_name = os.path.splitext(os.path.basename(elf_path))[0]

    instructions = disassemble(elf_path)
    trace = trace_isa_test(instructions, elf_path)
    stats = compute_timing(trace, config)

    # Instruction mix
    instr_mix = {}
    for instr in trace:
        kind_name = instr.kind.name.lower()
        instr_mix[kind_name] = instr_mix.get(kind_name, 0) + 1

    return {
        "test_name": test_name,
        "config": config,
        "methodology": "analytical",
        "dynamic_instruction_count": len(trace),
        "instruction_mix": instr_mix,
        "expected_stats": stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate performance reference statistics")
    parser.add_argument("elf", nargs="?", help="ELF file to analyze")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--batch", help="Directory of ELF files for batch processing")
    parser.add_argument("--output-dir", help="Output directory for batch mode")
    parser.add_argument("--trace", action="store_true",
                        help="Print dynamic instruction trace")
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)

    if args.batch:
        if not args.output_dir:
            print("Error: --output-dir required with --batch", file=sys.stderr)
            sys.exit(1)
        os.makedirs(args.output_dir, exist_ok=True)

        elf_files = sorted([
            os.path.join(args.batch, f)
            for f in os.listdir(args.batch)
            if os.path.isfile(os.path.join(args.batch, f))
            and not f.endswith(".o")
        ])

        for elf_path in elf_files:
            name = os.path.basename(elf_path)
            print(f"Analyzing {name}...", end=" ", flush=True)
            try:
                ref = generate_reference(elf_path, config)
                out_path = os.path.join(args.output_dir, f"{name}.json")
                with open(out_path, "w") as f:
                    json.dump(ref, f, indent=2)
                    f.write("\n")
                print(f"OK ({ref['dynamic_instruction_count']} instructions, "
                      f"{ref['expected_stats']['total_cycles']} cycles)")
            except Exception as e:
                print(f"FAILED: {e}")

    elif args.elf:
        ref = generate_reference(args.elf, config)

        if args.trace:
            instructions = disassemble(args.elf)
            trace = trace_isa_test(instructions, args.elf)
            for i, instr in enumerate(trace):
                print(f"  {i:4d}: {instr.addr:08x}  {instr.mnemonic:8s} {instr.operands:30s}  "
                      f"[{instr.kind.name:8s}]"
                      f"{'  rd=x' + str(instr.rd) if instr.has_rd else ''}"
                      f"{'  rs1=x' + str(instr.rs1) if instr.rs1 is not None else ''}"
                      f"{'  rs2=x' + str(instr.rs2) if instr.rs2 is not None else ''}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(ref, f, indent=2)
                f.write("\n")
            print(f"Written to {args.output}")
        else:
            print(json.dumps(ref, indent=2))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
