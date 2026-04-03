#pragma once

#include "gpu_sim/types.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace gpu_sim {

class FlatMemory {
public:
    explicit FlatMemory(uint32_t size_bytes)
        : data_(size_bytes, 0) {}

    uint32_t size() const { return static_cast<uint32_t>(data_.size()); }

    uint8_t read8(Address addr) const {
        check_bounds(addr, 1);
        return data_[addr];
    }

    uint16_t read16(Address addr) const {
        check_bounds(addr, 2);
        uint16_t val;
        std::memcpy(&val, &data_[addr], 2);
        return val;  // Little-endian
    }

    uint32_t read32(Address addr) const {
        check_bounds(addr, 4);
        uint32_t val;
        std::memcpy(&val, &data_[addr], 4);
        return val;  // Little-endian
    }

    void write8(Address addr, uint8_t val) {
        check_bounds(addr, 1);
        data_[addr] = val;
    }

    void write16(Address addr, uint16_t val) {
        check_bounds(addr, 2);
        std::memcpy(&data_[addr], &val, 2);
    }

    void write32(Address addr, uint32_t val) {
        check_bounds(addr, 4);
        std::memcpy(&data_[addr], &val, 4);
    }

    // Bulk access for loading binaries
    uint8_t* data() { return data_.data(); }
    const uint8_t* data() const { return data_.data(); }

private:
    void check_bounds(Address addr, uint32_t size) const {
        if (addr >= data_.size() || size > data_.size() - addr) {
            throw std::out_of_range("Memory access out of bounds: addr=" +
                std::to_string(addr) + " size=" + std::to_string(size));
        }
    }

    std::vector<uint8_t> data_;
};

class InstructionMemory {
public:
    explicit InstructionMemory(uint32_t size_bytes)
        : instructions_(size_bytes / 4, 0) {}

    uint32_t read(Address pc) const {
        uint32_t index = pc / 4;
        if (index >= instructions_.size()) {
            throw std::out_of_range("Instruction fetch out of bounds: pc=" + std::to_string(pc));
        }
        return instructions_[index];
    }

    void write(uint32_t index, uint32_t instruction) {
        if (index >= instructions_.size()) {
            throw std::out_of_range("Instruction write out of bounds: index=" + std::to_string(index));
        }
        instructions_[index] = instruction;
    }

    uint32_t num_instructions() const { return static_cast<uint32_t>(instructions_.size()); }

    // Bulk access for loading binaries
    uint32_t* data() { return instructions_.data(); }
    const uint32_t* data() const { return instructions_.data(); }
    uint32_t size_bytes() const { return static_cast<uint32_t>(instructions_.size() * 4); }

private:
    std::vector<uint32_t> instructions_;
};

class LookupTable {
public:
    explicit LookupTable(uint32_t num_entries)
        : entries_(num_entries, 0) {}

    uint32_t read(uint32_t index) const {
        if (index >= entries_.size()) {
            return 0;  // Out-of-bounds reads return 0
        }
        return entries_[index];
    }

    void write(uint32_t index, uint32_t value) {
        if (index < entries_.size()) {
            entries_[index] = value;
        }
    }

    uint32_t num_entries() const { return static_cast<uint32_t>(entries_.size()); }
    uint32_t* data() { return entries_.data(); }

private:
    std::vector<uint32_t> entries_;
};

} // namespace gpu_sim
