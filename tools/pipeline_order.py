"""Single source of truth for the timing-model pipeline dataflow order.

`MODULE_ORDER` lists every timing-model module in dataflow order, front
to back: frontend stages first, execution units, the memory subsystem,
the writeback arbiter, and the control-cluster orchestrators last.

This ordering is consumed by:

  - `diagram_extract_ast.py` / `diagram_extract_md.py` — module render
    order within and across clusters.
  - `render_signal_diagram.py` — back-pressure direction detection (an
    edge whose `src` index exceeds its `dst` index runs against the
    natural dataflow).
  - `lint_timing_naming.py` — the cross-module-read analysis pass
    classifies every `other->next_*()` read by dataflow direction:
    a read by an *upstream* module is a legitimate COMBINATIONAL
    back-pressure / stall edge; a read by a *downstream* module is a
    forbidden combinational-forward edge.

Keeping the order in one module guarantees the extractors, the renderer,
and the lint all share an identical notion of "upstream" / "downstream".
When a new module joins the pipeline, add it here once (the timing-
discipline "New module checklist" calls this out).

`CONTROL_MODULES` are the orchestrator modules (`PanicController`,
`TimingModel`). They sit at the end of `MODULE_ORDER` purely as a
layout sink — they are side-channel observers, not part of the linear
pipeline dataflow — so direction-based analyses (back-pressure overlay,
the lint's forward/backward classification) treat them specially rather
than as ordinary downstream stages.
"""

from __future__ import annotations

# Pipeline dataflow order, front to back. Frozen as a tuple so callers
# cannot mutate the shared list; callers that need a list build their own.
MODULE_ORDER: tuple[str, ...] = (
    "FetchStage",
    "DecodeStage",
    "WarpScheduler",
    "Scoreboard",
    "BranchShadowTracker",
    "OperandCollector",
    "ALUUnit",
    "MultiplyUnit",
    "DivideUnit",
    "TLookupUnit",
    "LdStUnit",
    "CoalescingUnit",
    "LoadGatherBufferFile",
    "L1Cache",
    "ExternalMemoryInterface",
    "WritebackArbiter",
    "PanicController",
    "TimingModel",
)

# Orchestrator modules. They observe arbitrary module state from
# orchestration methods (`TimingModel::tick`, `record_cycle_trace`,
# `PanicController::evaluate`) and are not part of the linear pipeline
# dataflow, so direction-based analyses exempt them.
CONTROL_MODULES: frozenset[str] = frozenset({"PanicController", "TimingModel"})


def module_index(name: str) -> int:
    """Return the dataflow index of `name`, or -1 if it is not a known
    pipeline module."""
    try:
        return MODULE_ORDER.index(name)
    except ValueError:
        return -1
