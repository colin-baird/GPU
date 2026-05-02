    .option norvc

    # jalr_storm: probes spec §4.2 JALR static-mispredict path + redirect
    # propagation. Each loop iteration executes one JALR whose register-
    # indirect target is NOT PC+4 (forces mispredict per the impl: predictor
    # default predicted_target=0, branch_mispredicted compares predicted_next
    # vs actual; mismatch → flush). Two filler instructions sit on the
    # predicted fall-through path; they should be flushed and never reach
    # writeback.
    #
    # Per iteration: 1 ALU (addi -1) + 1 JALR (mispredict) + 1 BNEZ (correct
    # except on final iter when counter reaches 0).
    #
    # ARG0 (x1) = iteration count.

    .section .text
    .globl _start

_start:
    mv   x12, x1            # iteration counter
    la   x20, target        # JALR target (resolves to PC of `target` label)

loop:
    addi x12, x12, -1
    jalr x0, x20, 0         # mispredicted (predicted PC+4, actual = target)
    addi x10, x10, 1        # speculative — flushed on mispredict
    addi x10, x10, 1        # ditto

target:
    bnez x12, loop          # backward conditional, predicted taken

    ecall
