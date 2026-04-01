#include "gpu_sim/timing/branch_predictor.h"
#include "gpu_sim/decoder.h"

namespace gpu_sim {

BranchPrediction StaticDirectionalBranchPredictor::predict(uint32_t pc,
                                                           uint32_t raw_instruction) const {
    const DecodedInstruction decoded = Decoder::decode(raw_instruction);
    BranchPrediction prediction;

    switch (decoded.type) {
        case InstructionType::BRANCH:
            prediction.is_control_flow = true;
            prediction.predicted_taken = decoded.imm < 0;
            prediction.predicted_target = pc + static_cast<uint32_t>(decoded.imm);
            break;

        case InstructionType::JAL:
            prediction.is_control_flow = true;
            prediction.predicted_taken = true;
            prediction.predicted_target = pc + static_cast<uint32_t>(decoded.imm);
            break;

        case InstructionType::JALR:
            prediction.is_control_flow = true;
            break;

        default:
            break;
    }

    return prediction;
}

void StaticDirectionalBranchPredictor::update(uint32_t pc, const DecodedInstruction& decoded,
                                              const BranchPrediction& prediction,
                                              bool actual_taken,
                                              uint32_t actual_target) {
    static_cast<void>(pc);
    static_cast<void>(decoded);
    static_cast<void>(prediction);
    static_cast<void>(actual_taken);
    static_cast<void>(actual_target);
}

} // namespace gpu_sim
