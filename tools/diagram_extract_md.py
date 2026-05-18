"""Markdown extractor for the signal-flow diagram.

Parses `resources/timing_discipline.md`'s "Per-boundary inventory" table
and returns an `ExtractionResult`. The AST extractor under
`diagram_extract_ast.py` is the primary source post-Phase 4; this module
is retained for `--source=markdown` and `--validate` cross-checks.
"""

from __future__ import annotations

import re
from pathlib import Path

from diagram_types import Edge, ExtractionResult, Module

# Canonical module name -> cluster label. Order within a cluster
# influences left-to-right layout in the rendered diagram.
MODULES: list[tuple[str, str]] = [
    # Frontend & Issue (in tick-order along the front of the pipeline).
    ("FetchStage",              "Frontend & Issue"),
    ("DecodeStage",             "Frontend & Issue"),
    ("WarpScheduler",           "Frontend & Issue"),
    ("Scoreboard",              "Frontend & Issue"),
    ("BranchShadowTracker",     "Frontend & Issue"),
    ("OperandCollector",        "Frontend & Issue"),
    # Execute units (siblings).
    ("ALUUnit",                 "Execute"),
    ("MultiplyUnit",            "Execute"),
    ("DivideUnit",              "Execute"),
    ("TLookupUnit",             "Execute"),
    ("LdStUnit",                "Execute"),
    # Memory subsystem.
    ("CoalescingUnit",          "Memory"),
    ("LoadGatherBufferFile",    "Memory"),
    ("L1Cache",                 "Memory"),
    ("ExternalMemoryInterface", "Memory"),
    # Writeback.
    ("WritebackArbiter",        "Writeback"),
    # Control / orchestration.
    ("PanicController",         "Control"),
    ("TimingModel",             "Control"),
]

# Alias patterns matched in producer/consumer cells. Order matters: the
# longest, most-specific names are tried first so e.g. "L1Cache" matches
# before the lowercase "cache" alias would. Patterns use word boundaries
# to avoid matching substrings of unrelated identifiers.
ALIAS_PATTERNS_RAW: list[tuple[str, str]] = [
    # Canonical PascalCase identifiers.
    (r"\bExternalMemoryInterface\b", "ExternalMemoryInterface"),
    (r"\bLoadGatherBufferFile\b",    "LoadGatherBufferFile"),
    (r"\bBranchShadowTracker\b",     "BranchShadowTracker"),
    (r"\bWritebackArbiter\b",        "WritebackArbiter"),
    (r"\bOperandCollector\b",        "OperandCollector"),
    (r"\bCoalescingUnit\b",          "CoalescingUnit"),
    (r"\bWarpScheduler\b",           "WarpScheduler"),
    (r"\bPanicController\b",         "PanicController"),
    (r"\bTLookupUnit\b",             "TLookupUnit"),
    (r"\bMultiplyUnit\b",            "MultiplyUnit"),
    (r"\bDivideUnit\b",              "DivideUnit"),
    (r"\bDecodeStage\b",             "DecodeStage"),
    (r"\bFetchStage\b",              "FetchStage"),
    (r"\bScoreboard\b",              "Scoreboard"),
    (r"\bTimingModel\b",             "TimingModel"),
    (r"\bL1Cache\b",                 "L1Cache"),
    (r"\bLdStUnit\b",                "LdStUnit"),
    (r"\bALUUnit\b",                 "ALUUnit"),
    # Lowercase / shorthand aliases used in prose and code. Snake-case
    # forms drop the trailing `\b` so member-access syntax like
    # `branch_tracker_.foo()` matches (the trailing `_` is a word char,
    # which would otherwise defeat `\b`).
    (r"\bbranch_tracker",            "BranchShadowTracker"),
    (r"\bgather_file",               "LoadGatherBufferFile"),
    (r"\bwb_arbiter",                "WritebackArbiter"),
    (r"\bcoalescing\b",              "CoalescingUnit"),
    (r"\bscheduler\b",               "WarpScheduler"),
    (r"\bopcoll\b",                  "OperandCollector"),
    (r"\bmem_if",                    "ExternalMemoryInterface"),
    (r"\btlookup\b",                 "TLookupUnit"),
    (r"\bdecode\b",                  "DecodeStage"),
    (r"\bfetch\b",                   "FetchStage"),
    (r"\bldst\b",                    "LdStUnit"),
    (r"\bcache\b",                   "L1Cache"),
    (r"\bALU\b",                     "ALUUnit"),
    (r"\bMUL\b",                     "MultiplyUnit"),
    (r"\bDIV\b",                     "DivideUnit"),
]
ALIAS_PATTERNS = [(re.compile(p, re.IGNORECASE), c) for p, c in ALIAS_PATTERNS_RAW]


# Modules that the prose pluralises into a generic group name. When the
# group token appears in a cell, the listed canonical members are emitted.
GROUP_FANOUT: dict[str, list[str]] = {
    "ExecutionUnit": ["ALUUnit", "MultiplyUnit", "DivideUnit", "TLookupUnit", "LdStUnit"],
}
GROUP_PATTERNS: list[tuple[re.Pattern[str], list[str]]] = [
    (re.compile(rf"\b{name}s?\b"), members) for name, members in GROUP_FANOUT.items()
]
GROUP_PATTERNS.append(
    (re.compile(r"\bexecution_units?\w*", re.IGNORECASE), GROUP_FANOUT["ExecutionUnit"]),
)

# Inventory rows that are genuinely internal-to-one-module (no
# cross-stage edge to draw). Listed here so --check stays clean.
#
#   - Row 6: OperandCollector's internal next_/current_ double-buffer.
#   - Row 3: the operand-collector-busy poll the warp scheduler used to
#     perform was replaced in Phase 10B.0 by an internal issue
#     scoreboard (`opcoll_cooldown_cycles_`). The scheduler no longer
#     reads `OperandCollector::current_busy()` cross-stage, so the row
#     documents scheduler-internal bookkeeping with no edge to draw.
INTERNAL_ROWS: set[int] = {3, 6}

# Default short signal name per inventory row, used as the visible edge
# label for non-overridden rows. Auto-extracted Payload text is too long
# and not consistently noun-like; these are 1-3 word handles aligned
# with the doc's prose.
ROW_LABEL: dict[int, str] = {
    1:  "busy",           # decode→fetch back-pressure
    2:  "pending",        # decode→fetch pending warp id
    3:  "opcoll busy",
    4:  "busy",           # unit→sched (5 edges)
    8:  "scoreboard",
    11: "gather port",
    12: "redirect",
    14: "drained",
    15: "mem req/resp",
}

# Some inventory rows compound multiple unrelated signals into one row,
# which makes the producer×consumer cross-product produce many false
# edges between modules that happen to co-occur in the prose. For those
# rows we hand-curate the canonical edges with explicit edge labels.
# Format: row_number -> list of (src, dst, classification, label).
ROW_OVERRIDES: dict[int, list[tuple[str, str, str, str]]] = {
    # Row 1: decode publishes current_busy() (REGISTERED, back-pressure
    # direction) and fetch publishes its committed output via
    # current_output() that decode reads in evaluate() (REGISTERED,
    # forward-data direction).
    1: [
        ("DecodeStage", "FetchStage",  "REGISTERED", "busy"),
        ("FetchStage",  "DecodeStage", "REGISTERED", "output"),
    ],
    # Row 3: no cross-stage edge. Phase 10B.0 replaced the scheduler's
    # poll of `OperandCollector::current_busy()` with an internal issue
    # scoreboard (`opcoll_cooldown_cycles_`); the operand-collector-busy
    # signal is no longer read across the stage boundary. The row stays
    # in the inventory for the historical record but emits no edge (see
    # also INTERNAL_ROWS).
    3: [],
    # Row 4: post-Phase-10B.0 the warp scheduler tracks fully-pipelined
    # and iterative execution-unit availability from its own issue
    # scoreboard (`unit_busy_` countdown), not by polling each unit's
    # `current_busy()`. The one surviving unit→scheduler edge is the
    # LdSt address-gen FIFO accounting (`current_fifo_size()` /
    # `current_fifo_total_pushes()` / `current_fifo_capacity()`), which
    # the scheduler reads to enforce the FIFO-slot bound at issue.
    4: [
        ("LdStUnit", "WarpScheduler", "REGISTERED", "fifo accounting"),
    ],
    # Row 5: per-warp branch-shadow bit. Three writers funnel into
    # BranchShadowTracker; scheduler reads the tracker. Branch
    # resolution moved into `ALUUnit::evaluate()` in Phase 10A, so the
    # correct-prediction writer is the ALU (not the operand collector).
    5: [
        ("WarpScheduler",       "BranchShadowTracker", "REGISTERED", "branch issued"),
        ("ALUUnit",             "BranchShadowTracker", "REGISTERED", "branch resolved"),
        ("FetchStage",          "BranchShadowTracker", "REGISTERED", "redirect applied"),
        ("BranchShadowTracker", "WarpScheduler",       "REGISTERED", "in_flight"),
    ],
    # Row 7: the REGISTERED issue/execute path. Phase 10B registered
    # every edge of it: scheduler→opcoll (10B.2), opcoll→unit (10B.1),
    # unit→wb_arbiter (10B.3). Each execution unit and the gather-buffer
    # file publish results via REGISTERED result_buffer slots that
    # WritebackArbiter consumes. The LdSt→Coalescing addr-gen FIFO edge
    # is REGISTERED as of Phase M1.
    7: [
        ("WarpScheduler",        "OperandCollector", "REGISTERED", "issue"),
        ("OperandCollector",     "ALUUnit",          "REGISTERED", "operands"),
        ("OperandCollector",     "MultiplyUnit",     "REGISTERED", "operands"),
        ("OperandCollector",     "DivideUnit",       "REGISTERED", "operands"),
        ("OperandCollector",     "TLookupUnit",      "REGISTERED", "operands"),
        ("OperandCollector",     "LdStUnit",         "REGISTERED", "operands"),
        ("ALUUnit",              "WritebackArbiter", "REGISTERED", "result"),
        ("MultiplyUnit",         "WritebackArbiter", "REGISTERED", "result"),
        ("DivideUnit",           "WritebackArbiter", "REGISTERED", "result"),
        ("TLookupUnit",          "WritebackArbiter", "REGISTERED", "result"),
        ("LdStUnit",             "WritebackArbiter", "REGISTERED", "result"),
        ("LoadGatherBufferFile", "WritebackArbiter", "REGISTERED", "result"),
        ("LdStUnit",             "CoalescingUnit",   "REGISTERED", "addr_gen FIFO"),
    ],
    # Row 8: scoreboard. Both WarpScheduler and WritebackArbiter write
    # via set_pending / clear_pending; scheduler reads is_pending.
    8: [
        ("Scoreboard",       "WarpScheduler", "REGISTERED", "scoreboard"),
        ("WarpScheduler",    "Scoreboard",    "REGISTERED", "scoreboard"),
        ("WritebackArbiter", "Scoreboard",    "REGISTERED", "scoreboard"),
    ],
    # Row 9: CoalescingUnit drives three downstream subsystems mid-tick;
    # L1Cache stall flows back same-cycle COMBINATIONAL; gather-buffer
    # busyness is a REGISTERED back-pressure accessor read at the top of
    # coalescing's evaluate.
    9: [
        ("CoalescingUnit",       "LdStUnit",             "REGISTERED",    "FIFO pop"),
        ("CoalescingUnit",       "LoadGatherBufferFile", "REGISTERED",    "claim"),
        ("CoalescingUnit",       "L1Cache",              "REGISTERED",    "load/store"),
        ("L1Cache",              "CoalescingUnit",       "COMBINATIONAL", "stalled"),
        ("LoadGatherBufferFile", "CoalescingUnit",       "REGISTERED",    "busy"),
    ],
    # Row 10: cache external surface. record_cycle_trace reads
    # registered scratch; CoalescingUnit reads the COMBINATIONAL stall
    # signal (already covered by row 9, deduped at emit time).
    10: [
        ("L1Cache", "TimingModel",    "REGISTERED",    "trace events"),
        ("L1Cache", "CoalescingUnit", "COMBINATIONAL", "stalled"),
    ],
    # Row 12: misprediction redirect. `ALUUnit::evaluate` asserts the
    # `next_redirect_` transient (branch resolution lives in the ALU
    # since Phase 10A); `FetchStage::evaluate` and `DecodeStage::evaluate`
    # read `alu->next_redirect()` the same cycle and apply the flush.
    # COMBINATIONAL backward (Phase 10E converted the prior REGISTERED
    # ALU-staged form to the discipline-correct combinational-backward
    # transient).
    12: [
        ("ALUUnit", "FetchStage",  "COMBINATIONAL", "redirect"),
        ("ALUUnit", "DecodeStage", "COMBINATIONAL", "redirect"),
    ],
    # Row 13 mixes two distinct signals: (a) DecodeStage publishes the
    # EBREAK request that TimingModel reads at top-of-tick, and (b) the
    # panic-flush cascade where PanicController triggers flush() on
    # scheduler / opcoll / gather buffer / writeback arbiter at the
    # commit-phase boundary.
    13: [
        ("DecodeStage",     "TimingModel",          "REGISTERED", "ebreak"),
        ("PanicController", "WarpScheduler",        "REGISTERED", "flush"),
        ("PanicController", "OperandCollector",     "REGISTERED", "flush"),
        ("PanicController", "LoadGatherBufferFile", "REGISTERED", "flush"),
        ("PanicController", "WritebackArbiter",     "REGISTERED", "flush"),
    ],
    # Row 14: drained-query callable. The lambda reads current_busy() /
    # next_has_result() / next_fifo_empty() on each execution unit plus
    # opcoll and wb_arbiter — all const accessors over committed state
    # (REGISTERED with back-pressure direction overlay).
    14: [
        ("ALUUnit",              "PanicController", "REGISTERED", "drained"),
        ("MultiplyUnit",         "PanicController", "REGISTERED", "drained"),
        ("DivideUnit",           "PanicController", "REGISTERED", "drained"),
        ("TLookupUnit",          "PanicController", "REGISTERED", "drained"),
        ("LdStUnit",             "PanicController", "REGISTERED", "drained"),
        ("OperandCollector",     "PanicController", "REGISTERED", "drained"),
        ("WritebackArbiter",     "PanicController", "REGISTERED", "drained"),
    ],
    # Row 15: cache↔mem_if. L1Cache submits requests (REGISTERED writes
    # into mem_if's queue); cache reads current_has_response() /
    # get_response() — a REGISTERED poll over committed end-of-cycle
    # state (Phase M5 renamed next_has_response → current_has_response).
    15: [
        ("L1Cache",                 "ExternalMemoryInterface", "REGISTERED", "mem request"),
        ("ExternalMemoryInterface", "L1Cache",                 "REGISTERED", "has_response"),
    ],
    # Row 16: the writeback stall. `WritebackArbiter::evaluate` asserts
    # the COMBINATIONAL `next_writeback_stall()` transient when a
    # variable-latency load preempts a fixed-latency unit's writeback.
    # The five execution units and the operand collector read it at the
    # top of their `commit()` (gating the next_*→current_* flip); the
    # warp scheduler reads it at the top of `evaluate()` (gating issue).
    # COMBINATIONAL backward / control — the arbiter is sequenced first
    # in the evaluate sweep so every upstream consumer reads it
    # same-cycle.
    16: [
        ("WritebackArbiter", "ALUUnit",          "COMBINATIONAL", "writeback stall"),
        ("WritebackArbiter", "MultiplyUnit",     "COMBINATIONAL", "writeback stall"),
        ("WritebackArbiter", "DivideUnit",       "COMBINATIONAL", "writeback stall"),
        ("WritebackArbiter", "TLookupUnit",      "COMBINATIONAL", "writeback stall"),
        ("WritebackArbiter", "LdStUnit",         "COMBINATIONAL", "writeback stall"),
        ("WritebackArbiter", "OperandCollector", "COMBINATIONAL", "writeback stall"),
        ("WritebackArbiter", "WarpScheduler",    "COMBINATIONAL", "writeback stall"),
    ],
}


def _find_modules(cell: str) -> list[str]:
    """Return canonical module names mentioned in a cell, deduped, in
    first-occurrence order."""
    hits: list[tuple[int, str]] = []
    for pat, canonical in ALIAS_PATTERNS:
        for m in pat.finditer(cell):
            hits.append((m.start(), canonical))
    for pat, members in GROUP_PATTERNS:
        for m in pat.finditer(cell):
            for member in members:
                hits.append((m.start(), member))
    hits.sort()
    out: list[str] = []
    seen: set[str] = set()
    for _, canonical in hits:
        if canonical not in seen:
            out.append(canonical)
            seen.add(canonical)
    return out


def _parse_inventory(md: str) -> list[dict]:
    """Extract rows from the per-boundary inventory markdown table."""
    rows: list[dict] = []
    lines = md.splitlines()
    header: list[str] | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if (
            header is None
            and line.lstrip().startswith("| #")
            and "Producer" in line
            and "Consumer" in line
        ):
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2  # skip the |---|---| separator
            continue
        if header is not None:
            stripped = line.strip()
            if not stripped.startswith("|"):
                # A blank line inside the inventory table does not end
                # it: the table may be visually broken by a spacer line
                # (e.g. an appended row separated for readability). Only
                # a non-blank, non-pipe line truly closes the table.
                if not stripped:
                    # Look ahead past consecutive blank lines: if the
                    # next non-blank line is another table row, the
                    # blank line is an intra-table spacer; keep the
                    # header active.
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    if j < len(lines) and lines[j].lstrip().startswith("|"):
                        i += 1
                        continue
                header = None
                i += 1
                continue
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if cells and cells[0].isdigit():
                if len(cells) < len(header):
                    cells = cells + [""] * (len(header) - len(cells))
                row = dict(zip(header, cells))
                row["__full__"] = stripped
                row["row_number"] = int(cells[0])
                rows.append(row)
        i += 1
    return rows


def _classify(row: dict) -> list[str]:
    """Classifications mentioned in a row.

    Under the two-axis model the classification axis carries only the
    cycle discipline (REGISTERED or COMBINATIONAL). Back-pressure used
    to be a third value (READY/STALL); it is now an architectural
    direction overlay derived from cluster topology by
    `tools/render_signal_diagram.py` rather than a signal class.

    Prefers the dedicated `Cycle` column (post Phase 0 rewrite), then
    falls back to the legacy `Classification` cell, and finally scans
    the whole row prose.
    """
    primary = row.get("Cycle", "") or row.get("Classification", "")
    primary = primary.lower()
    found: list[str] = []
    if "registered" in primary:
        found.append("REGISTERED")
    if "combinational" in primary:
        found.append("COMBINATIONAL")
    if found:
        return found
    full = row.get("__full__", "").lower()
    if "registered" in full or "next_" in full:
        found.append("REGISTERED")
    if "combinational" in full:
        found.append("COMBINATIONAL")
    return found or ["UNKNOWN"]


def _short_payload(s: str) -> str:
    """Compress a payload cell for use as an edge label."""
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 48:
        s = s[:45] + "..."
    return s


def _edges_for_row(row: dict) -> list[Edge]:
    """Generate styled edges for one inventory row.

    Strategy: take the cross-product of canonical modules mentioned in
    the Producer cell vs. the Consumer cell. Self-edges and edges that
    duplicate an earlier (src, dst, class) tuple within the same row are
    suppressed. A few rows with many writers (e.g. row 5's branch
    tracker fan-in) over-generate edges, but those over-generated edges
    are still architecturally truthful.
    """
    row_num = row["row_number"]
    default_label = ROW_LABEL.get(row_num) or _short_payload(row.get("Payload", ""))

    if row_num in ROW_OVERRIDES:
        return [
            Edge(src=s, dst=d, classification=c, label=lbl, source_row=row_num)
            for (s, d, c, lbl) in ROW_OVERRIDES[row_num]
        ]

    producers = _find_modules(row.get("Producer", ""))
    consumers = _find_modules(row.get("Consumer", ""))
    classes = _classify(row)
    primary = classes[0]

    seen: set[tuple[str, str, str]] = set()
    edges: list[Edge] = []
    for src in producers:
        for dst in consumers:
            if src == dst:
                continue
            key = (src, dst, primary)
            if key in seen:
                continue
            seen.add(key)
            edges.append(Edge(
                src=src,
                dst=dst,
                classification=primary,
                label=default_label,
                source_row=row_num,
            ))
    return edges


def extract(md_path: Path) -> ExtractionResult:
    """Parse the timing-discipline markdown into modules + edges."""
    md = md_path.read_text()
    rows = _parse_inventory(md)
    if not rows:
        return ExtractionResult(
            modules=[Module(name=n, cluster=c) for (n, c) in MODULES],
            edges=[],
            warnings=[f"no inventory table found in {md_path}"],
        )

    all_edges: list[Edge] = []
    empty_rows: list[int] = []
    unknown_rows: list[int] = []
    for row in rows:
        row_edges = _edges_for_row(row)
        if not row_edges and row["row_number"] not in INTERNAL_ROWS:
            empty_rows.append(row["row_number"])
        if _classify(row) == ["UNKNOWN"]:
            unknown_rows.append(row["row_number"])
        all_edges.extend(row_edges)

    warnings: list[str] = [f"parsed {len(rows)} inventory rows -> {len(all_edges)} edges"]
    if empty_rows:
        warnings.append(f"rows with no extractable edges: {empty_rows}")
    if unknown_rows:
        warnings.append(f"rows with unrecognised classification: {unknown_rows}")

    return ExtractionResult(
        modules=[Module(name=n, cluster=c) for (n, c) in MODULES],
        edges=all_edges,
        warnings=warnings,
    )
