#include "gpu_sim/elf_loader.h"
#include <fstream>
#include <vector>
#include <cstring>
#include <stdexcept>

// Minimal ELF32 definitions (portable, no dependency on system <elf.h>)
namespace {

static constexpr uint8_t ELFMAG0 = 0x7F;
static constexpr uint8_t ELFMAG1 = 'E';
static constexpr uint8_t ELFMAG2 = 'L';
static constexpr uint8_t ELFMAG3 = 'F';
static constexpr uint8_t ELFCLASS32 = 1;
static constexpr uint8_t ELFDATA2LSB = 1;
static constexpr uint16_t EM_RISCV = 243;
static constexpr uint32_t PT_LOAD = 1;

#pragma pack(push, 1)
struct Elf32_Ehdr {
    uint8_t  e_ident[16];
    uint16_t e_type;
    uint16_t e_machine;
    uint32_t e_version;
    uint32_t e_entry;
    uint32_t e_phoff;
    uint32_t e_shoff;
    uint32_t e_flags;
    uint16_t e_ehsize;
    uint16_t e_phentsize;
    uint16_t e_phnum;
    uint16_t e_shentsize;
    uint16_t e_shnum;
    uint16_t e_shstrndx;
};

struct Elf32_Phdr {
    uint32_t p_type;
    uint32_t p_offset;
    uint32_t p_vaddr;
    uint32_t p_paddr;
    uint32_t p_filesz;
    uint32_t p_memsz;
    uint32_t p_flags;
    uint32_t p_align;
};
#pragma pack(pop)

bool is_elf(const std::vector<uint8_t>& data) {
    if (data.size() < sizeof(Elf32_Ehdr)) return false;
    return data[0] == ELFMAG0 && data[1] == ELFMAG1 &&
           data[2] == ELFMAG2 && data[3] == ELFMAG3;
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    auto size = file.tellg();
    file.seekg(0);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

} // anonymous namespace

namespace gpu_sim {

ProgramImage load_program_image(const std::string& path) {
    auto data = read_file(path);
    ProgramImage image;

    if (is_elf(data)) {
        Elf32_Ehdr ehdr;
        std::memcpy(&ehdr, data.data(), sizeof(ehdr));

        if (ehdr.e_ident[4] != ELFCLASS32) {
            throw std::runtime_error("ELF: not 32-bit");
        }
        if (ehdr.e_ident[5] != ELFDATA2LSB) {
            throw std::runtime_error("ELF: not little-endian");
        }
        if (ehdr.e_machine != EM_RISCV) {
            throw std::runtime_error("ELF: not RISC-V (e_machine=" +
                std::to_string(ehdr.e_machine) + ")");
        }

        image.entry_pc = ehdr.e_entry;

        for (uint16_t i = 0; i < ehdr.e_phnum; ++i) {
            Elf32_Phdr phdr;
            size_t offset = ehdr.e_phoff + i * ehdr.e_phentsize;
            if (offset + sizeof(phdr) > data.size()) break;
            std::memcpy(&phdr, data.data() + offset, sizeof(phdr));

            if (phdr.p_type != PT_LOAD) continue;
            if (phdr.p_filesz == 0) continue;

            ProgramSegment seg;
            seg.vaddr = phdr.p_vaddr;
            seg.data.resize(phdr.p_filesz);
            std::memcpy(seg.data.data(), data.data() + phdr.p_offset,
                        phdr.p_filesz);
            image.segments.push_back(std::move(seg));
        }
    } else {
        // Raw binary: single segment at address 0
        ProgramSegment seg;
        seg.vaddr = 0;
        seg.data = std::move(data);
        image.segments.push_back(std::move(seg));
        image.entry_pc = 0;
    }

    return image;
}

void load_image_into_model(FunctionalModel& model, const ProgramImage& image) {
    uint32_t instr_mem_end = model.instruction_memory().size_bytes();

    for (const auto& seg : image.segments) {
        if (seg.vaddr < instr_mem_end) {
            // Load into instruction memory
            uint32_t copy_size = std::min(static_cast<uint32_t>(seg.data.size()),
                instr_mem_end - seg.vaddr);
            uint32_t num_instrs = copy_size / 4;
            for (uint32_t j = 0; j < num_instrs; ++j) {
                uint32_t instr;
                std::memcpy(&instr, seg.data.data() + j * 4, 4);
                model.instruction_memory().write(seg.vaddr / 4 + j, instr);
            }
        } else {
            // Load into data memory
            uint32_t copy_size = static_cast<uint32_t>(seg.data.size());
            if (seg.vaddr + copy_size <= model.memory().size()) {
                std::memcpy(model.memory().data() + seg.vaddr,
                            seg.data.data(), copy_size);
            }
        }
    }
}

uint32_t load_program(FunctionalModel& model, const std::string& path) {
    ProgramImage image = load_program_image(path);
    load_image_into_model(model, image);
    return image.entry_pc;
}

void load_lookup_table(FunctionalModel& model, const std::string& path) {
    auto data = read_file(path);
    uint32_t num_entries = static_cast<uint32_t>(data.size()) / 4;
    uint32_t max_entries = model.lookup_table().num_entries();
    num_entries = std::min(num_entries, max_entries);

    for (uint32_t i = 0; i < num_entries; ++i) {
        uint32_t val;
        std::memcpy(&val, data.data() + i * 4, 4);
        model.lookup_table().write(i, val);
    }
}

void load_data(FunctionalModel& model, const std::string& path, uint32_t base_addr) {
    auto data = read_file(path);
    uint32_t copy_size = static_cast<uint32_t>(data.size());
    if (base_addr + copy_size > model.memory().size()) {
        copy_size = model.memory().size() - base_addr;
    }
    std::memcpy(model.memory().data() + base_addr, data.data(), copy_size);
}

} // namespace gpu_sim
