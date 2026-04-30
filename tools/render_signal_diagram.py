#!/usr/bin/env python3
"""Render the timing-model signal-flow architecture poster.

Reads `resources/timing_discipline.md`, parses the per-boundary inventory
table, extracts cross-stage edges between simulator modules, and emits:

  - A Graphviz DOT file (`tools/signal_diagram.dot`) — the architecture
    poster, with modules grouped into Frontend & Issue / Execute / Memory /
    Writeback / Control clusters and edges styled by classification
    (REGISTERED solid / COMBINATIONAL dashed / READY-STALL dotted).
  - A Mermaid companion (`tools/signal_diagram.mmd`) for inline embedding
    in markdown viewers.

If Graphviz `dot` is on PATH and `--svg` is passed, the DOT is rendered to
`tools/signal_diagram.svg` as well.

The discipline document is the source of truth for boundary classifications
(per project convention). This tool does not read C++ sources directly.

Usage:
    python3 tools/render_signal_diagram.py
    python3 tools/render_signal_diagram.py --svg
    python3 tools/render_signal_diagram.py --check    # lint, no output files
"""

import argparse
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "resources" / "timing_discipline.md"
DEFAULT_OUT_DIR = REPO_ROOT / "tools"

# Canonical module name -> cluster label. Order within a cluster is
# preserved in the DOT output to influence layout left-to-right along
# the natural data-flow order.
MODULES = [
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
    # Control / orchestration. Outside the four primary clusters but
    # included because the doc classifies edges into these nodes.
    ("PanicController",         "Control"),
    ("TimingModel",             "Control"),
]
MODULE_CLUSTER = dict(MODULES)
CLUSTER_ORDER = ["Frontend & Issue", "Execute", "Memory", "Writeback", "Control"]

# Multi-line node labels — PascalCase split so boxes can be near-square
# without truncating the longer names.
NODE_LABEL = {
    "FetchStage":              "Fetch\\nStage",
    "DecodeStage":             "Decode\\nStage",
    "WarpScheduler":           "Warp\\nScheduler",
    "Scoreboard":              "Scoreboard",
    "BranchShadowTracker":     "Branch\\nShadow\\nTracker",
    "OperandCollector":        "Operand\\nCollector",
    "ALUUnit":                 "ALU\\nUnit",
    "MultiplyUnit":            "Multiply\\nUnit",
    "DivideUnit":              "Divide\\nUnit",
    "TLookupUnit":             "TLookup\\nUnit",
    "LdStUnit":                "LdSt\\nUnit",
    "CoalescingUnit":          "Coalescing\\nUnit",
    "LoadGatherBufferFile":    "LoadGather\\nBufferFile",
    "L1Cache":                 "L1\\nCache",
    "ExternalMemoryInterface": "External\\nMemory\\nInterface",
    "WritebackArbiter":        "Writeback\\nArbiter",
    "PanicController":         "Panic\\nController",
    "TimingModel":             "Timing\\nModel",
}
CLUSTER_COLOR = {
    "Frontend & Issue": "#dbeafe",
    "Execute":          "#fef3c7",
    "Memory":           "#dcfce7",
    "Writeback":        "#fee2e2",
    "Control":          "#ede9fe",
}

# Alias patterns matched in producer/consumer cells. Order matters: the
# longest, most-specific names are tried first so that e.g. "L1Cache"
# matches before the lowercase "cache" alias would. Patterns use word
# boundaries to avoid matching substrings of unrelated identifiers.
ALIAS_PATTERNS_RAW = [
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
    # forms drop the trailing `\b` so that member-access syntax like
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
GROUP_FANOUT = {
    "ExecutionUnit": ["ALUUnit", "MultiplyUnit", "DivideUnit", "TLookupUnit", "LdStUnit"],
}
GROUP_PATTERNS = [(re.compile(rf"\b{name}s?\b"), members) for name, members in GROUP_FANOUT.items()]
# Prose-form fanouts (snake_case, possibly embedded in identifiers).
GROUP_PATTERNS.append(
    (re.compile(r"\bexecution_units?\w*", re.IGNORECASE), GROUP_FANOUT["ExecutionUnit"]),
)

# Inventory rows that are genuinely internal-to-one-module (no
# cross-stage edge to draw). Listed here so --check stays clean.
INTERNAL_ROWS = {6}

# Default short signal name per inventory row, used as the visible edge
# label for non-overridden rows. Auto-extracted Payload text is too long
# and not consistently noun-like; these are 1-3 word handles aligned
# with the doc's prose.
ROW_LABEL = {
    1:  "ready",          # decode→fetch back-pressure
    2:  "pending",        # decode→fetch pending warp id
    3:  "opcoll ready",
    4:  "ready",          # unit→sched (5 edges)
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
    # Row 5: per-warp branch-shadow bit. Three writers all funnel into
    # BranchShadowTracker; scheduler reads the tracker.
    5: [
        ("WarpScheduler",       "BranchShadowTracker", "REGISTERED", "branch issued"),
        ("OperandCollector",    "BranchShadowTracker", "REGISTERED", "branch resolved"),
        ("FetchStage",          "BranchShadowTracker", "REGISTERED", "redirect applied"),
        ("BranchShadowTracker", "WarpScheduler",       "REGISTERED", "in_flight"),
    ],
    # Row 7: each execution unit publishes its result via a REGISTERED
    # result_buffer that WritebackArbiter consumes (with a same-tick
    # has_result COMBINATIONAL probe). Plus the LdSt→Coalescing addr-gen
    # edge is COMBINATIONAL.
    7: [
        ("ALUUnit",      "WritebackArbiter", "REGISTERED",    "result"),
        ("MultiplyUnit", "WritebackArbiter", "REGISTERED",    "result"),
        ("DivideUnit",   "WritebackArbiter", "REGISTERED",    "result"),
        ("TLookupUnit",  "WritebackArbiter", "REGISTERED",    "result"),
        ("LdStUnit",     "WritebackArbiter", "REGISTERED",    "result"),
        ("LdStUnit",     "CoalescingUnit",   "COMBINATIONAL", "addr_gen FIFO"),
    ],
    # Row 9: CoalescingUnit drives three downstream subsystems mid-tick;
    # L1Cache stall flows back same-cycle COMBINATIONAL.
    9: [
        ("CoalescingUnit", "LdStUnit",             "REGISTERED",    "FIFO pop"),
        ("CoalescingUnit", "LoadGatherBufferFile", "REGISTERED",    "claim"),
        ("CoalescingUnit", "L1Cache",              "REGISTERED",    "load/store"),
        ("L1Cache",        "CoalescingUnit",       "COMBINATIONAL", "stalled"),
    ],
    # Row 10: cache external surface. record_cycle_trace reads
    # registered scratch; CoalescingUnit reads the COMBINATIONAL stall
    # signal (already covered by row 9, deduped at emit time).
    10: [
        ("L1Cache", "TimingModel",    "REGISTERED",    "trace events"),
        ("L1Cache", "CoalescingUnit", "COMBINATIONAL", "stalled"),
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
}


def find_modules(cell: str) -> list[str]:
    """Return the canonical module names mentioned in a cell, deduped,
    in first-occurrence order."""
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


def parse_inventory(md: str) -> list[dict]:
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
                header = None
                i += 1
                continue
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if cells and cells[0].isdigit():
                # Some rows in the doc collapse the tick-order column into
                # an adjacent cell, producing fewer cells than the header
                # has columns. Pad with empties so dict access is safe.
                if len(cells) < len(header):
                    cells = cells + [""] * (len(header) - len(cells))
                row = dict(zip(header, cells))
                # Stash the full row as a single string for whole-row keyword
                # fallbacks (e.g. classification heuristics).
                row["__full__"] = stripped
                row["row_number"] = int(cells[0])
                rows.append(row)
        i += 1
    return rows


def classify(row: dict) -> list[str]:
    """Classifications mentioned in a row.

    Prefers the dedicated Classification cell, but falls back to scanning
    the whole row when the cell lacks the keyword (a few rows in the doc
    use the Classification cell for free-form notes and label the actual
    discipline elsewhere — e.g. row 11's `next_*` arbitration is described
    in Current state, row 14's wired-callable semantics are in Refactor
    phase prose).
    """
    primary = row.get("Classification", "").lower()
    found: list[str] = []
    if "registered" in primary:
        found.append("REGISTERED")
    if "combinational" in primary:
        found.append("COMBINATIONAL")
    if "ready/stall" in primary:
        found.append("READY/STALL")
    if found:
        return found
    full = row.get("__full__", "").lower()
    if "registered" in full or "next_" in full:
        found.append("REGISTERED")
    if "combinational" in full:
        found.append("COMBINATIONAL")
    if "ready/stall" in full or "ready_out" in full:
        found.append("READY/STALL")
    return found or ["UNKNOWN"]


def short_payload(s: str) -> str:
    """Compress a payload cell for use as an edge label."""
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 48:
        s = s[:45] + "..."
    return s


def edges_for_row(row: dict) -> list[dict]:
    """Generate styled edges for one inventory row.

    Strategy: take the cross-product of canonical modules mentioned in
    the Producer cell vs. the Consumer cell. Self-edges and edges that
    duplicate an earlier (src, dst, class) tuple within the same row are
    suppressed. This is a deliberate first-cut: a few rows with many
    writers (e.g. row 5's branch tracker fan-in) over-generate edges,
    but those over-generated edges are still architecturally truthful
    (the named modules really do touch each other in that row).
    """
    row_num = row["row_number"]
    default_label = ROW_LABEL.get(row_num) or short_payload(row.get("Payload", ""))

    if row_num in ROW_OVERRIDES:
        return [
            {"src": s, "dst": d, "classification": c, "row": row_num, "label": lbl}
            for (s, d, c, lbl) in ROW_OVERRIDES[row_num]
        ]

    producers = find_modules(row.get("Producer", ""))
    consumers = find_modules(row.get("Consumer", ""))
    classes = classify(row)
    primary = classes[0]

    seen: set[tuple[str, str, str]] = set()
    edges: list[dict] = []
    for src in producers:
        for dst in consumers:
            if src == dst:
                continue
            key = (src, dst, primary)
            if key in seen:
                continue
            seen.add(key)
            edges.append({
                "src": src,
                "dst": dst,
                "classification": primary,
                "row": row_num,
                "label": default_label,
            })
    return edges


# ----------------------------------------------------------------------------
# Output emitters
# ----------------------------------------------------------------------------

EDGE_STYLE = {
    "REGISTERED":    {"color": "#1f2937", "style": "solid"},
    "COMBINATIONAL": {"color": "#d97706", "style": "dashed"},
    "READY/STALL":   {"color": "#2563eb", "style": "dotted", "arrowhead": "empty"},
    "UNKNOWN":       {"color": "#9ca3af", "style": "dashed"},
}


def emit_dot(edges: list[dict]) -> str:
    out: list[str] = []
    out.append("digraph signal_flow {")
    out.append('  rankdir=TB;')
    # Title and a small inline legend live in an HTML graph label so the
    # legend doesn't form a disconnected component (which used to confuse
    # graphviz's rank assignment).
    legend_html = (
        '<<table border="0" cellborder="0" cellspacing="2">'
        '<tr><td align="center" colspan="3"><font point-size="14"><b>Timing-model signal flow</b></font><br/>'
        '<font point-size="9">generated from resources/timing_discipline.md</font></td></tr>'
        '<tr>'
        '<td><font color="#1f2937">━━</font> REGISTERED (1-cycle)</td>'
        '<td><font color="#d97706">┄┄</font> COMBINATIONAL (same-cycle)</td>'
        '<td><font color="#2563eb">┈┈▷</font> READY/STALL (back-pressure)</td>'
        '</tr></table>>'
    )
    out.append(
        f'  graph [fontname="Helvetica", label={legend_html}, labelloc=t, pad=0.3, '
        'ranksep=0.7, nodesep=0.6, splines=ortho, clusterrank=local, compound=true, '
        # esep adds minimum gap around each edge. With ortho routing,
        # parallel edges share corridors; +12 keeps parallel runs
        # (5 unit→Panic "drained", 5 unit→Sched "ready", etc.) visibly
        # separated.
        'esep="+12", '
        # forcelabels=true lets graphviz move xlabels (external edge
        # labels) to avoid collisions with other elements. This is the
        # graphviz-recommended approach for dense ortho graphs.
        'forcelabels=true];'
    )
    out.append('  node  [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", fillcolor="#ffffff", color="#374151", width=1.4, height=1.1];')
    out.append('  edge  [fontname="Helvetica", fontsize=10, color="#374151"];')
    out.append("")

    by_cluster: dict[str, list[str]] = defaultdict(list)
    for name, cluster in MODULES:
        by_cluster[cluster].append(name)

    for ci, cluster in enumerate(CLUSTER_ORDER):
        if not by_cluster.get(cluster):
            continue
        sanitized = cluster.replace(" ", "_").replace("&", "and")
        out.append(f'  subgraph cluster_{sanitized} {{')
        # HTML label with an opaque white background so the cluster
        # title stays readable when ortho edges happen to pass through
        # the cluster's title row.
        cluster_label_html = cluster.replace("&", "&amp;")
        cluster_html = (
            f'<<table border="0" cellborder="0" cellpadding="3">'
            f'<tr><td bgcolor="white"><b>{cluster_label_html}</b></td></tr></table>>'
        )
        out.append(f'    label={cluster_html};')
        # Title at the bottom keeps it clear of incoming edges that
        # cross the top of the cluster.
        out.append('    labelloc=b;')
        out.append('    labeljust=c;')
        out.append('    style="filled,rounded";')
        out.append(f'    fillcolor="{CLUSTER_COLOR[cluster]}";')
        out.append('    color="#9ca3af";')
        out.append('    fontsize=13;')
        # Larger margin pushes content away from the cluster boundary
        # so the title (which sits at the boundary) has clear space
        # above/below the nearest edge traffic.
        out.append('    margin=20;')
        for name in by_cluster[cluster]:
            label = NODE_LABEL.get(name, name)
            out.append(f'    "{name}" [label="{label}"];')
        out.append("  }")
        out.append("")


    # Edges. Group by (src, dst, classification) and aggregate row numbers.
    # Within a group, distinct labels are joined with " / ".
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for e in edges:
        grouped[(e["src"], e["dst"], e["classification"])].append(e)

    # Per-node fan counts (used to choose head/tail label placement: put
    # labels at whichever endpoint is the "spread" side, so labels stay
    # near unique anchors instead of stacking at a shared endpoint).
    fan_out: dict[str, int] = defaultdict(int)
    fan_in: dict[str, int] = defaultdict(int)
    for (s, d, _) in grouped:
        fan_out[s] += 1
        fan_in[d] += 1

    for (src, dst, klass), group in sorted(grouped.items()):
        style = EDGE_STYLE[klass]
        sorted_group = sorted(group, key=lambda g: g["row"])
        rows_label = ",".join(str(g["row"]) for g in sorted_group)
        labels_seen: list[str] = []
        for g in sorted_group:
            if g["label"] and g["label"] not in labels_seen:
                labels_seen.append(g["label"])
        edge_label = " / ".join(labels_seen) if labels_seen else f"row {rows_label}"
        attrs = [f'color="{style["color"]}"', f'style={style["style"]}']
        if "arrowhead" in style:
            attrs.append(f'arrowhead={style["arrowhead"]}')
        # constraint=false marks edges that shouldn't influence rank
        # assignment, so layout follows the main data-flow DAG and these
        # edges float around it.
        # - READY/STALL: flow backward against data direction.
        # - Control-cluster edges (PanicController, TimingModel): a
        #   side-channel observer; letting these edges constrain ranks
        #   pulls TimingModel away from PanicController and stretches
        #   the Control cluster across the whole graph height.
        if (klass == "READY/STALL"
                or MODULE_CLUSTER.get(src) == "Control"
                or MODULE_CLUSTER.get(dst) == "Control"):
            attrs.append("constraint=false")
        # xlabel = external auxiliary label. With forcelabels=true on
        # the graph, graphviz repositions xlabels along the edge to
        # avoid collisions with nodes, cluster boxes, and other labels.
        # This works better than head/tail anchoring for dense graphs.
        attrs.append(f'xlabel="{edge_label}"')
        attrs.append(f'tooltip="row {rows_label}: {edge_label}"')
        out.append(f'  "{src}" -> "{dst}" [{", ".join(attrs)}];')

    out.append("}")
    return "\n".join(out) + "\n"


def emit_mermaid(edges: list[dict]) -> str:
    out: list[str] = []
    out.append("flowchart LR")
    by_cluster: dict[str, list[str]] = defaultdict(list)
    for name, cluster in MODULES:
        by_cluster[cluster].append(name)
    for cluster in CLUSTER_ORDER:
        if not by_cluster.get(cluster):
            continue
        cid = cluster.replace(" ", "_").replace("&", "and")
        out.append(f'  subgraph {cid}["{cluster}"]')
        for name in by_cluster[cluster]:
            out.append(f"    {name}")
        out.append("  end")

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for e in edges:
        grouped[(e["src"], e["dst"], e["classification"])].append(e)
    for (src, dst, klass), group in sorted(grouped.items()):
        sorted_group = sorted(group, key=lambda g: g["row"])
        labels_seen: list[str] = []
        for g in sorted_group:
            if g["label"] and g["label"] not in labels_seen:
                labels_seen.append(g["label"])
        edge_label = " / ".join(labels_seen) if labels_seen else "row " + ",".join(
            str(g["row"]) for g in sorted_group
        )
        # Mermaid requires escaping pipe and quotes in labels.
        edge_label = edge_label.replace("|", "\\|").replace('"', "'")
        if klass == "REGISTERED" or klass == "UNKNOWN":
            out.append(f"  {src} -->|{edge_label}| {dst}")
        elif klass == "COMBINATIONAL":
            out.append(f"  {src} -.->|{edge_label}| {dst}")
        else:  # READY/STALL
            out.append(f"  {src} -. {edge_label} .-> {dst}")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                    help="Path to timing_discipline.md (default: %(default)s)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="Output directory (default: %(default)s)")
    ap.add_argument("--svg", action="store_true",
                    help="Also render an SVG via `dot` if available")
    ap.add_argument("--check", action="store_true",
                    help="Lint only: report rows that produced no edges and exit nonzero on findings")
    args = ap.parse_args(argv)

    md = args.input.read_text()
    rows = parse_inventory(md)
    if not rows:
        print(f"error: no inventory table found in {args.input}", file=sys.stderr)
        return 2

    all_edges: list[dict] = []
    empty_rows: list[int] = []
    unknown_rows: list[int] = []
    for row in rows:
        edges = edges_for_row(row)
        if not edges and row["row_number"] not in INTERNAL_ROWS:
            empty_rows.append(row["row_number"])
        if classify(row) == ["UNKNOWN"]:
            unknown_rows.append(row["row_number"])
        all_edges.extend(edges)

    print(f"parsed {len(rows)} inventory rows -> {len(all_edges)} edges")
    if empty_rows:
        print(f"warning: rows with no extractable edges: {empty_rows}", file=sys.stderr)
    if unknown_rows:
        print(f"warning: rows with unrecognised classification: {unknown_rows}", file=sys.stderr)

    if args.check:
        return 1 if (empty_rows or unknown_rows) else 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dot_path = args.out_dir / "signal_diagram.dot"
    mmd_path = args.out_dir / "signal_diagram.mmd"
    dot_path.write_text(emit_dot(all_edges))
    mmd_path.write_text(emit_mermaid(all_edges))
    print(f"wrote {dot_path}")
    print(f"wrote {mmd_path}")

    if args.svg:
        if shutil.which("dot") is None:
            print("warning: `dot` not found on PATH; skipping SVG render", file=sys.stderr)
        else:
            svg_path = args.out_dir / "signal_diagram.svg"
            # `dot` may emit non-fatal init_rank warnings on densely
            # clustered graphs and exit nonzero even though it wrote the
            # SVG. Treat the file's existence as the success signal.
            result = subprocess.run(
                ["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)],
                capture_output=True, text=True,
            )
            if svg_path.exists() and svg_path.stat().st_size > 0:
                print(f"wrote {svg_path}")
                if result.returncode != 0:
                    print(f"note: dot reported a warning (exit {result.returncode}): {result.stderr.strip()}", file=sys.stderr)
            else:
                print(f"error: dot failed (exit {result.returncode}): {result.stderr}", file=sys.stderr)
                return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
