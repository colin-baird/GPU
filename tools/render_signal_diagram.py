#!/usr/bin/env python3
"""Render the timing-model signal-flow architecture poster.

Drives the architecture poster from one of two sources:

  - **Markdown extractor** (`--source=markdown`): parses
    `resources/timing_discipline.md`'s per-boundary inventory table.
    Authored by hand; the cross-check view of the same data.
  - **AST extractor** (`--source=ast`, default in Phase 4): walks the
    libclang AST of the timing-model C++ sources, listed in
    `build/compile_commands.json`. The C++ source is the source of truth.

Both extractors produce `Module` and `Edge` records (see
`diagram_types.py`); this script emits Graphviz DOT, Mermaid, a draw.io
file, and optionally SVG/PNG from those records.

The DOT poster is a fixed 2-D layout with hand-pinned node positions,
rendered by Graphviz `neato` (not `dot`) — see `emit_dot` / `MODULE_GRID`.

Output files (under `tools/` by default):
  - `signal_diagram.dot` — Graphviz source (neato, pinned positions).
  - `signal_diagram.mmd` — Mermaid companion for inline embedding.
  - `signal_diagram.drawio` — hand-editable draw.io / diagrams.net file:
    same layout, but with draggable boxes and orthogonal connectors.
  - `signal_diagram.svg` / `.png` — only when `--svg` is passed and
    `neato` is on PATH.

Usage:
    python3 tools/render_signal_diagram.py
    python3 tools/render_signal_diagram.py --svg
    python3 tools/render_signal_diagram.py --check       # lint, no output files
    python3 tools/render_signal_diagram.py --validate    # diff AST vs markdown
    python3 tools/render_signal_diagram.py --source=ast  # AST extractor
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Ensure sibling modules import cleanly when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagram_types import Edge, ExtractionResult, Module  # noqa: E402
import diagram_extract_md  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "resources" / "timing_discipline.md"
DEFAULT_OUT_DIR = REPO_ROOT / "tools"
DEFAULT_COMPILE_DB = REPO_ROOT / "build" / "compile_commands.json"
DEFAULT_SIM_ROOT = REPO_ROOT / "sim"

# Cluster emission order. With the pinned-position layout the order is
# purely cosmetic (it sets the order of subgraph blocks in the .dot and
# the Mermaid companion); node placement is fixed by MODULE_GRID below.
CLUSTER_ORDER = ["Control", "Frontend & Issue", "Execute", "Memory", "Writeback"]

# ----------------------------------------------------------------------------
# Poster layout grid
# ----------------------------------------------------------------------------
# The diagram is a fixed 2-D poster, rendered with neato in pinned-
# position mode — every node carries an explicit `pos="x,y!"`. graphviz's
# `dot` rank model cannot express this arrangement (a full-width Control
# band *above* clusters that are themselves side-by-side), so the layout
# is pinned by hand here and neato is left only to route the edges and
# draw the cluster boxes.
#
# Coordinates are (column, row): column increases rightward, row
# increases downward; both are scaled to inches by COL_SPACING /
# ROW_SPACING when emitted. Row 1 is deliberately left empty as breathing
# room between the Control band and the dataflow below it.
#
#   - Control       — a thin full-width band across the top.
#   - Frontend & Issue — a 2x3 block on the left.
#   - Execute       — the five units as a single vertical bank.
#   - Memory        — a horizontal pipeline to the right of Execute.
#   - Writeback     — tucked directly below Memory.
MODULE_GRID: dict[str, tuple[float, float]] = {
    "PanicController":         (2.0, 0.0),
    "TimingModel":             (5.0, 0.0),
    "FetchStage":              (0.0, 2.0),
    "BranchShadowTracker":     (1.0, 2.0),
    "DecodeStage":             (0.0, 3.0),
    "Scoreboard":              (1.0, 3.0),
    "WarpScheduler":           (0.0, 4.0),
    "OperandCollector":        (1.0, 4.0),
    "ALUUnit":                 (2.0, 2.0),
    "MultiplyUnit":            (2.0, 3.0),
    "DivideUnit":              (2.0, 4.0),
    "TLookupUnit":             (2.0, 5.0),
    "LdStUnit":                (2.0, 6.0),
    "CoalescingUnit":          (3.0, 3.0),
    "L1Cache":                 (4.0, 3.0),
    "ExternalMemoryInterface": (5.0, 3.0),
    "LoadGatherBufferFile":    (4.0, 4.0),
    "WritebackArbiter":        (4.0, 5.5),
}

COL_SPACING = 3.80    # inches between grid columns
ROW_SPACING = 2.45    # inches between grid rows
GRID_MAX_ROW = 6.0    # bottom-most row index, used to flip to neato's y-up axis

# Cluster-box sizing. neato sizes a cluster box to its member nodes, but
# with pinned positions its bbox under-reaches and clips the boxes. To
# get consistent, predictable boxes each cluster instead carries two
# invisible "extent anchor" nodes pinned at the desired corners; neato's
# bbox then expands to include them. Values are inches. The bottom pad is
# larger to reserve room for the bottom-anchored cluster label.
NODE_HALF_W = 0.70
NODE_HALF_H = 0.55
CLUSTER_PAD_X = 0.34
CLUSTER_PAD_TOP = 0.30
CLUSTER_PAD_BOTTOM = 0.62

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

# Cycle-discipline styling: solid = REGISTERED, dashed = COMBINATIONAL.
# The back-pressure direction overlay (dotted-blue) is applied at render
# time based on cluster topology — see `_apply_direction_overlay`. It is
# orthogonal to the cycle axis: a back-pressure REGISTERED accessor and
# a back-pressure COMBINATIONAL accessor both get the dotted-blue
# overlay, but keep their cycle-axis line style.
EDGE_STYLE = {
    "REGISTERED":    {"color": "#1f2937", "style": "solid"},
    "COMBINATIONAL": {"color": "#d97706", "style": "dashed"},
    "UNKNOWN":       {"color": "#9ca3af", "style": "dashed"},
}

# Override style applied when an edge is detected as back-pressure
# (consumer→producer, against the dataflow direction). Color and the
# empty arrowhead come from the prior READY/STALL style; the line style
# is preserved from the cycle-axis classification so REGISTERED stays
# solid-blue (committed-state read) and COMBINATIONAL stays
# dashed-blue (live mid-tick read).
BACK_PRESSURE_OVERRIDE = {"color": "#2563eb", "arrowhead": "empty"}

# Module dataflow ordering used to detect back-pressure. Edges whose
# `src` index is greater than `dst` index in this list run against the
# natural dataflow and render with the back-pressure overlay.
MODULE_DATAFLOW_ORDER: list[str] = [
    "FetchStage", "DecodeStage", "WarpScheduler", "Scoreboard",
    "BranchShadowTracker", "OperandCollector",
    "ALUUnit", "MultiplyUnit", "DivideUnit", "TLookupUnit", "LdStUnit",
    "CoalescingUnit", "LoadGatherBufferFile", "L1Cache",
    "ExternalMemoryInterface", "WritebackArbiter",
    "PanicController", "TimingModel",
]


def _is_back_pressure(src: str, dst: str) -> bool:
    """True if an edge runs against the natural dataflow (consumer
    reading producer's back-pressure signal). Excludes edges to/from
    Control-cluster modules (PanicController, TimingModel) which are
    side-channel observers, not part of the linear dataflow."""
    if src in ("PanicController", "TimingModel"):
        return False
    if dst in ("PanicController", "TimingModel"):
        return False
    try:
        si = MODULE_DATAFLOW_ORDER.index(src)
        di = MODULE_DATAFLOW_ORDER.index(dst)
    except ValueError:
        return False
    return si > di

# Validate-mode allow-list. Reserved for future divergences that are
# legitimate (e.g. a markdown row that documents an architectural
# carve-out the AST can't infer); empty as of the AST/markdown
# reconciliation pass.
VALIDATE_ALLOW_LIST: set[tuple[str, str, str, str]] = set()


# ----------------------------------------------------------------------------
# Output emitters
# ----------------------------------------------------------------------------

SOURCE_LEGEND = {
    "ast":      "generated from sim/{include,src}/timing/ via libclang AST",
    "markdown": "generated from resources/timing_discipline.md",
}


def emit_dot(modules: list[Module], edges: list[Edge],
             *, source: str = "ast") -> str:
    """Emit a Graphviz DOT representation of the diagram.

    The layout is a fixed 2-D poster (see ``MODULE_GRID``): Frontend &
    Issue on the left, the Execute units as a single vertical bank, the
    Memory pipeline to the right, Writeback directly below Memory, and
    Control as a thin band across the top. Every node carries an explicit
    pinned ``pos``; the file declares ``layout=neato`` so the positions
    are honoured and neato is left only to route the edges and draw the
    cluster boxes. Render with neato (``neato -Tsvg signal_diagram.dot``);
    plain ``dot`` also works since it honours the ``layout`` attribute.
    """
    out: list[str] = []
    out.append("digraph signal_flow {")
    # Pinned-position layout: neato honours each node's `pos="x,y!"`.
    # Declaring the engine in-file means a bare `dot signal_diagram.dot`
    # picks neato too.
    out.append('  layout="neato";')
    # Title and a small inline legend live in an HTML graph label.
    subtitle = SOURCE_LEGEND.get(source, SOURCE_LEGEND["ast"])
    legend_html = (
        '<<table border="0" cellborder="0" cellspacing="2">'
        '<tr><td align="center" colspan="3"><font point-size="14"><b>Timing-model signal flow</b></font><br/>'
        f'<font point-size="9">{subtitle}</font></td></tr>'
        '<tr>'
        '<td><font color="#1f2937">━━</font> REGISTERED (1-cycle)</td>'
        '<td><font color="#d97706">┄┄</font> COMBINATIONAL (same-cycle)</td>'
        '<td><font color="#2563eb">▷</font> back-pressure direction overlay</td>'
        '</tr></table>>'
    )
    out.append(
        f'  graph [fontname="Helvetica", label={legend_html}, labelloc=t, pad=0.4, '
        # splines=polyline routes each edge as its own straight-segment
        # path. Unlike splines=ortho (which packs parallel wires ~3px
        # apart into a shared lane with no knob to spread them), polyline
        # fans the edges out around each node's perimeter so dense hubs
        # like PanicController stay readable. sep/esep enlarge the
        # keep-clear margin around each node, spreading the fan further.
        # outputorder=edgesfirst paints the node boxes on top of the
        # edges so a box stays clean where an edge runs close.
        'splines=polyline, outputorder=edgesfirst, sep="+18", esep="+13", '
        # forcelabels=true lets graphviz move xlabels (external edge
        # labels) to avoid collisions with other elements.
        'forcelabels=true];'
    )
    # fixedsize=true keeps every box at the declared 1.4x1.1in, so the
    # MODULE_GRID spacing is predictable regardless of label length.
    out.append('  node  [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", fillcolor="#ffffff", color="#374151", width=1.4, height=1.1, fixedsize=true];')
    out.append('  edge  [fontname="Helvetica", fontsize=10, color="#374151"];')
    out.append("")

    def _pos(grid: tuple[float, float]) -> str:
        """Map a (column, row) grid cell to a pinned neato `pos` string.
        Rows are flipped against GRID_MAX_ROW because neato's y-axis
        points up while the grid counts rows downward."""
        col, row = grid
        x = col * COL_SPACING
        y = (GRID_MAX_ROW - row) * ROW_SPACING
        return f"{x:.3f},{y:.3f}!"

    def _extent_anchors(cluster: str, members: list[str]
                        ) -> list[tuple[str, str]]:
        """Two invisible corner nodes that force the cluster's bounding
        box out to a consistent margin (see CLUSTER_PAD_*). Returns
        (node-name, pos) pairs for the top-left and bottom-right corners.
        The Control band is widened to the poster's left edge."""
        cols = [MODULE_GRID[n][0] for n in members]
        rows = [MODULE_GRID[n][1] for n in members]
        min_col = 0.0 if cluster == "Control" else min(cols)
        max_col, min_row, max_row = max(cols), min(rows), max(rows)
        left = min_col * COL_SPACING - NODE_HALF_W - CLUSTER_PAD_X
        right = max_col * COL_SPACING + NODE_HALF_W + CLUSTER_PAD_X
        top = (GRID_MAX_ROW - min_row) * ROW_SPACING + NODE_HALF_H + CLUSTER_PAD_TOP
        bot = (GRID_MAX_ROW - max_row) * ROW_SPACING - NODE_HALF_H - CLUSTER_PAD_BOTTOM
        tag = cluster.replace(" ", "_").replace("&", "and")
        return [
            (f"__ext_{tag}_tl", f"{left:.3f},{top:.3f}!"),
            (f"__ext_{tag}_br", f"{right:.3f},{bot:.3f}!"),
        ]

    by_cluster: dict[str, list[str]] = defaultdict(list)
    for m in modules:
        by_cluster[m.cluster].append(m.name)

    for cluster in CLUSTER_ORDER:
        if not by_cluster.get(cluster):
            continue
        sanitized = cluster.replace(" ", "_").replace("&", "and")
        out.append(f'  subgraph cluster_{sanitized} {{')
        # HTML label with an opaque white background so the cluster
        # title stays readable when an edge passes near it.
        cluster_label_html = cluster.replace("&", "&amp;")
        cluster_html = (
            f'<<table border="0" cellborder="0" cellpadding="3">'
            f'<tr><td bgcolor="white"><b>{cluster_label_html}</b></td></tr></table>>'
        )
        out.append(f'    label={cluster_html};')
        out.append('    labelloc=b;')
        out.append('    labeljust=c;')
        out.append('    style="filled,rounded";')
        out.append(f'    fillcolor="{CLUSTER_COLOR[cluster]}";')
        out.append('    color="#9ca3af";')
        out.append('    fontsize=13;')
        for name in by_cluster[cluster]:
            label = NODE_LABEL.get(name, name)
            out.append(
                f'    "{name}" [label="{label}", pos="{_pos(MODULE_GRID[name])}"];'
            )
        # Invisible corner anchors pin the cluster box to a consistent
        # margin; without them neato's bbox clips the member boxes.
        for anchor, anchor_pos in _extent_anchors(cluster, by_cluster[cluster]):
            out.append(
                f'    "{anchor}" [style=invis, width=0.01, height=0.01, '
                f'label="", pos="{anchor_pos}"];'
            )
        out.append("  }")
        out.append("")

    # Group edges by (src, dst, classification) so parallel rows collapse
    # onto a single rendered edge with combined labels.
    grouped: dict[tuple[str, str, str], list[Edge]] = defaultdict(list)
    for e in edges:
        grouped[(e.src, e.dst, e.classification)].append(e)

    for (src, dst, klass), group in sorted(grouped.items()):
        style = dict(EDGE_STYLE.get(klass, EDGE_STYLE["UNKNOWN"]))
        # Apply the back-pressure direction overlay when the edge runs
        # consumer→producer (against the natural dataflow ordering in
        # `MODULE_DATAFLOW_ORDER`). The overlay tints the color and
        # switches the arrowhead while preserving the cycle-axis line
        # style — so a back-pressure REGISTERED edge is solid blue and
        # a back-pressure COMBINATIONAL edge is dashed blue.
        back_pressure = _is_back_pressure(src, dst)
        if back_pressure:
            style.update(BACK_PRESSURE_OVERRIDE)
        sorted_group = sorted(group, key=lambda g: (g.source_row if g.source_row is not None else 0))
        rows_label = ",".join(
            str(g.source_row) for g in sorted_group if g.source_row is not None
        )
        labels_seen: list[str] = []
        for g in sorted_group:
            if g.label and g.label not in labels_seen:
                labels_seen.append(g.label)
        edge_label = " / ".join(labels_seen) if labels_seen else (
            f"row {rows_label}" if rows_label else ""
        )
        attrs = [f'color="{style["color"]}"', f'style={style["style"]}']
        if "arrowhead" in style:
            attrs.append(f'arrowhead={style["arrowhead"]}')
        # xlabel = external auxiliary label. With forcelabels=true on
        # the graph, graphviz repositions xlabels along the edge to
        # avoid collisions with nodes, cluster boxes, and other labels.
        # This works better than head/tail anchoring for dense graphs.
        attrs.append(f'xlabel="{edge_label}"')
        tooltip = f"row {rows_label}: {edge_label}" if rows_label else edge_label
        attrs.append(f'tooltip="{tooltip}"')
        out.append(f'  "{src}" -> "{dst}" [{", ".join(attrs)}];')

    out.append("}")
    return "\n".join(out) + "\n"


def emit_mermaid(modules: list[Module], edges: list[Edge]) -> str:
    """Emit a Mermaid flowchart for inline embedding."""
    out: list[str] = []
    out.append("flowchart LR")
    by_cluster: dict[str, list[str]] = defaultdict(list)
    for m in modules:
        by_cluster[m.cluster].append(m.name)
    for cluster in CLUSTER_ORDER:
        if not by_cluster.get(cluster):
            continue
        cid = cluster.replace(" ", "_").replace("&", "and")
        out.append(f'  subgraph {cid}["{cluster}"]')
        for name in by_cluster[cluster]:
            out.append(f"    {name}")
        out.append("  end")

    grouped: dict[tuple[str, str, str], list[Edge]] = defaultdict(list)
    for e in edges:
        grouped[(e.src, e.dst, e.classification)].append(e)
    for (src, dst, klass), group in sorted(grouped.items()):
        sorted_group = sorted(group, key=lambda g: (g.source_row if g.source_row is not None else 0))
        labels_seen: list[str] = []
        for g in sorted_group:
            if g.label and g.label not in labels_seen:
                labels_seen.append(g.label)
        if labels_seen:
            edge_label = " / ".join(labels_seen)
        else:
            rows = [str(g.source_row) for g in sorted_group if g.source_row is not None]
            edge_label = "row " + ",".join(rows) if rows else ""
        # Mermaid requires escaping pipe and quotes in labels.
        edge_label = edge_label.replace("|", "\\|").replace('"', "'")
        # Mermaid doesn't have a great way to overlay direction tints, so
        # the cycle axis dictates arrow style: solid (REGISTERED) /
        # dashed (COMBINATIONAL). Back-pressure direction is implicit in
        # the (src, dst) pair when read alongside the cluster layout.
        if klass == "COMBINATIONAL":
            out.append(f"  {src} -.->|{edge_label}| {dst}")
        else:
            out.append(f"  {src} -->|{edge_label}| {dst}")
    return "\n".join(out) + "\n"


# draw.io (mxGraph) geometry, in pixels. The diagram reuses MODULE_GRID
# but scales it to draw.io's y-down pixel space; the constants only need
# to be proportional and produce comfortably-sized boxes.
DRAWIO_COL = 280.0      # px between grid columns
DRAWIO_ROW = 185.0      # px between grid rows
DRAWIO_NODE_W = 132.0   # node box width
DRAWIO_NODE_H = 94.0    # node box height
DRAWIO_PAD_X = 26.0     # cluster-box padding, sides
DRAWIO_PAD_TOP = 22.0   # cluster-box padding, top
DRAWIO_PAD_BOTTOM = 48.0  # cluster-box padding, bottom (room for the label)


def _xml_escape(text: str) -> str:
    """Escape text for an XML attribute value."""
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))


def emit_drawio(modules: list[Module], edges: list[Edge],
                *, source: str = "ast") -> str:
    """Emit a draw.io / diagrams.net file (mxGraph XML).

    Unlike the DOT/SVG poster — a finished render — this is a hand-
    editable baseline. Nodes are pre-placed in the MODULE_GRID layout and
    edges are real orthogonal connectors bound to the node shapes
    (`source`/`target`), so in draw.io a box can be dragged (its wires
    re-route), a wire can be re-routed by dragging waypoints, and spacing
    is free. Open `signal_diagram.drawio` at app.diagrams.net, the
    desktop app, or the VS Code draw.io extension.
    """
    by_cluster: dict[str, list[str]] = defaultdict(list)
    for m in modules:
        by_cluster[m.cluster].append(m.name)

    def _center(name: str) -> tuple[float, float]:
        col, row = MODULE_GRID[name]
        return col * DRAWIO_COL, row * DRAWIO_ROW

    # Absolute (pre-shift) rectangles for nodes and cluster backgrounds.
    node_rect: dict[str, tuple[float, float, float, float]] = {}
    for m in modules:
        cx, cy = _center(m.name)
        node_rect[m.name] = (cx - DRAWIO_NODE_W/2, cy - DRAWIO_NODE_H/2,
                             DRAWIO_NODE_W, DRAWIO_NODE_H)

    cluster_rect: dict[str, tuple[float, float, float, float]] = {}
    for cluster in CLUSTER_ORDER:
        members = by_cluster.get(cluster)
        if not members:
            continue
        cxs = [_center(n)[0] for n in members]
        cys = [_center(n)[1] for n in members]
        # The Control band is stretched to the poster's left edge.
        min_x = (0.0 if cluster == "Control" else min(cxs))
        x0 = min_x - DRAWIO_NODE_W/2 - DRAWIO_PAD_X
        x1 = max(cxs) + DRAWIO_NODE_W/2 + DRAWIO_PAD_X
        y0 = min(cys) - DRAWIO_NODE_H/2 - DRAWIO_PAD_TOP
        y1 = max(cys) + DRAWIO_NODE_H/2 + DRAWIO_PAD_BOTTOM
        cluster_rect[cluster] = (x0, y0, x1 - x0, y1 - y0)

    # Shift everything so the top-left of the drawing sits at (20, 20).
    rects = list(node_rect.values()) + list(cluster_rect.values())
    off_x = 20.0 - min(r[0] for r in rects)
    off_y = 20.0 - min(r[1] for r in rects)
    page_w = max(r[0] + r[2] for r in rects) + off_x + 20.0
    page_h = max(r[1] + r[3] for r in rects) + off_y + 20.0

    out: list[str] = []
    out.append('<mxfile host="render_signal_diagram.py">')
    out.append('  <diagram name="Timing-model signal flow" id="signal-flow">')
    out.append(
        f'    <mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" '
        f'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" '
        f'pageScale="1" pageWidth="{page_w:.0f}" pageHeight="{page_h:.0f}" '
        f'math="0" shadow="0">'
    )
    out.append('      <root>')
    out.append('        <mxCell id="0" />')
    out.append('        <mxCell id="1" parent="0" />')

    # Cluster backgrounds first so they paint behind the node boxes.
    for cluster in CLUSTER_ORDER:
        if cluster not in cluster_rect:
            continue
        x, y, w, h = cluster_rect[cluster]
        cid = "cluster_" + cluster.replace(" ", "_").replace("&", "and")
        style = (f"rounded=1;arcSize=4;fillColor={CLUSTER_COLOR[cluster]};"
                 "strokeColor=#9ca3af;verticalAlign=bottom;align=center;"
                 "fontSize=15;fontStyle=1;fontColor=#374151;html=1;")
        out.append(f'        <mxCell id="{cid}" value="{_xml_escape(cluster)}" '
                   f'style="{style}" vertex="1" parent="1">')
        out.append(f'          <mxGeometry x="{x+off_x:.0f}" y="{y+off_y:.0f}" '
                   f'width="{w:.0f}" height="{h:.0f}" as="geometry" />')
        out.append('        </mxCell>')

    # Node boxes.
    for m in modules:
        x, y, w, h = node_rect[m.name]
        parts = NODE_LABEL.get(m.name, m.name).split("\\n")
        # The cell value is an XML attribute, so the <br> separators must
        # themselves be escaped; draw.io unescapes, then renders the HTML.
        value = "&lt;br&gt;".join(_xml_escape(p) for p in parts)
        style = ("rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;"
                 "strokeColor=#374151;fontSize=11;fontColor=#111827;")
        out.append(f'        <mxCell id="{m.name}" value="{value}" '
                   f'style="{style}" vertex="1" parent="1">')
        out.append(f'          <mxGeometry x="{x+off_x:.0f}" y="{y+off_y:.0f}" '
                   f'width="{w:.0f}" height="{h:.0f}" as="geometry" />')
        out.append('        </mxCell>')

    # Orthogonal connectors, grouped and styled like the DOT poster.
    grouped: dict[tuple[str, str, str], list[Edge]] = defaultdict(list)
    for e in edges:
        grouped[(e.src, e.dst, e.classification)].append(e)
    for idx, ((src, dst, klass), group) in enumerate(sorted(grouped.items())):
        base = EDGE_STYLE.get(klass, EDGE_STYLE["UNKNOWN"])
        back_pressure = _is_back_pressure(src, dst)
        stroke = BACK_PRESSURE_OVERRIDE["color"] if back_pressure else base["color"]
        sorted_group = sorted(
            group, key=lambda g: (g.source_row if g.source_row is not None else 0))
        labels_seen: list[str] = []
        for g in sorted_group:
            if g.label and g.label not in labels_seen:
                labels_seen.append(g.label)
        rows = [str(g.source_row) for g in sorted_group if g.source_row is not None]
        label = " / ".join(labels_seen) if labels_seen else (
            "row " + ",".join(rows) if rows else "")
        style = (f"edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;"
                 f"strokeColor={stroke};fontSize=10;fontColor=#374151;"
                 "labelBackgroundColor=#ffffff;jettySize=auto;")
        if base["style"] == "dashed":
            style += "dashed=1;"
        # Back-pressure wires keep the prior READY/STALL open arrowhead.
        if back_pressure:
            style += "endArrow=open;"
        out.append(f'        <mxCell id="e{idx}" value="{_xml_escape(label)}" '
                   f'style="{style}" edge="1" parent="1" '
                   f'source="{src}" target="{dst}">')
        out.append('          <mxGeometry relative="1" as="geometry" />')
        out.append('        </mxCell>')

    out.append('      </root>')
    out.append('    </mxGraphModel>')
    out.append('  </diagram>')
    out.append('</mxfile>')
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------
# Extraction dispatch
# ----------------------------------------------------------------------------

def run_extractor(source: str, *, md_path: Path, compile_db: Path,
                  sim_root: Path) -> ExtractionResult:
    """Dispatch to the requested extractor and return its result."""
    if source == "markdown":
        return diagram_extract_md.extract(md_path)
    if source == "ast":
        try:
            import diagram_extract_ast  # noqa: WPS433 — lazy import
        except ImportError as exc:
            return ExtractionResult(
                errors=[
                    f"AST extractor unavailable ({exc}). Install via "
                    "`pip install -r tools/requirements.txt` or rerun "
                    "with --source=markdown."
                ],
            )
        try:
            return diagram_extract_ast.extract(
                compile_commands=compile_db,
                sim_root=sim_root,
            )
        except (FileNotFoundError, ValueError) as exc:
            return ExtractionResult(errors=[str(exc)])
    raise ValueError(f"unknown source: {source!r}")


def print_extraction_errors(source: str, result: ExtractionResult) -> None:
    for err in result.errors:
        print(f"error: {source} extractor: {err}", file=sys.stderr)


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------

def diff_results(ast: ExtractionResult, md: ExtractionResult,
                 *, strict: bool = False) -> tuple[bool, list[str]]:
    """Compare two extraction results.

    Returns (had_differences, lines_to_print). Edges are compared on the
    (src, dst, classification) triple; labels are ignored because the
    extractors generate them differently. When `strict=False` the
    `VALIDATE_ALLOW_LIST` is honored — listed differences are reported
    as informational and don't flip `had_differences`.
    """
    lines: list[str] = []
    had_diff = False

    ast_modules = {m.name for m in ast.modules}
    md_modules = {m.name for m in md.modules}
    only_ast_mods = sorted(ast_modules - md_modules)
    only_md_mods = sorted(md_modules - ast_modules)
    if only_ast_mods:
        had_diff = True
        lines.append(f"modules only in AST: {only_ast_mods}")
    if only_md_mods:
        had_diff = True
        lines.append(f"modules only in markdown: {only_md_mods}")

    ast_edges = {(e.src, e.dst, e.classification) for e in ast.edges}
    md_edges = {(e.src, e.dst, e.classification) for e in md.edges}

    def _split_allowed(diffs: set[tuple[str, str, str]], side: str
                       ) -> tuple[list, list]:
        unexpected, allowed = [], []
        for triple in sorted(diffs):
            key = (*triple, side)
            (allowed if key in VALIDATE_ALLOW_LIST else unexpected).append(triple)
        return unexpected, allowed

    only_ast = ast_edges - md_edges
    only_md = md_edges - ast_edges
    ast_unexpected, ast_allowed = _split_allowed(only_ast, "ast-only")
    md_unexpected, md_allowed = _split_allowed(only_md, "markdown-only")

    if ast_unexpected:
        had_diff = True
        lines.append(f"edges only in AST ({len(ast_unexpected)}):")
        for triple in ast_unexpected:
            lines.append(f"  {triple[0]} -> {triple[1]} [{triple[2]}]")
    if ast_allowed and strict:
        had_diff = True
        lines.append(f"allow-listed AST-only edges ({len(ast_allowed)}):")
        for triple in ast_allowed:
            lines.append(f"  {triple[0]} -> {triple[1]} [{triple[2]}]")
    elif ast_allowed:
        lines.append(f"allow-listed AST-only edges ({len(ast_allowed)}, ignored):")
        for triple in ast_allowed:
            lines.append(f"  {triple[0]} -> {triple[1]} [{triple[2]}]")

    if md_unexpected:
        had_diff = True
        lines.append(f"edges only in markdown ({len(md_unexpected)}):")
        for triple in md_unexpected:
            lines.append(f"  {triple[0]} -> {triple[1]} [{triple[2]}]")
    if md_allowed and strict:
        had_diff = True
        lines.append(f"allow-listed markdown-only edges ({len(md_allowed)}):")
        for triple in md_allowed:
            lines.append(f"  {triple[0]} -> {triple[1]} [{triple[2]}]")
    elif md_allowed:
        lines.append(f"allow-listed markdown-only edges ({len(md_allowed)}, ignored):")
        for triple in md_allowed:
            lines.append(f"  {triple[0]} -> {triple[1]} [{triple[2]}]")

    # Classification disagreements: same (src, dst), different class.
    ast_pairs: dict[tuple[str, str], set[str]] = defaultdict(set)
    md_pairs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for e in ast.edges:
        ast_pairs[(e.src, e.dst)].add(e.classification)
    for e in md.edges:
        md_pairs[(e.src, e.dst)].add(e.classification)
    disagreements: list[tuple[tuple[str, str], set[str], set[str]]] = []
    for pair in sorted(set(ast_pairs) & set(md_pairs)):
        if ast_pairs[pair] != md_pairs[pair]:
            disagreements.append((pair, ast_pairs[pair], md_pairs[pair]))
    if disagreements:
        had_diff = True
        lines.append(f"classification disagreements ({len(disagreements)}):")
        for (src, dst), a, m in disagreements:
            lines.append(f"  {src} -> {dst}: ast={sorted(a)} markdown={sorted(m)}")

    return had_diff, lines


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                    help="Path to timing_discipline.md (default: %(default)s)")
    ap.add_argument("--compile-db", type=Path, default=DEFAULT_COMPILE_DB,
                    help="Path to compile_commands.json for the AST extractor "
                         "(default: %(default)s)")
    ap.add_argument("--sim-root", type=Path, default=DEFAULT_SIM_ROOT,
                    help="Path to the sim source root (default: %(default)s)")
    ap.add_argument("--source", choices=("markdown", "ast"), default="ast",
                    help="Extractor to use (default: %(default)s)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="Output directory (default: %(default)s)")
    ap.add_argument("--svg", action="store_true",
                    help="Also render signal_diagram.svg and .png via "
                         "`neato` if available")
    ap.add_argument("--check", action="store_true",
                    help="Lint only: report extractor warnings and exit nonzero")
    ap.add_argument("--validate", action="store_true",
                    help="Run both extractors and diff their results; exit "
                         "nonzero on any difference (the allow-list at "
                         "render_signal_diagram.VALIDATE_ALLOW_LIST is "
                         "empty by default — both sources should agree)")
    ap.add_argument("--strict", action="store_true",
                    help="With --validate, also fail on allow-listed "
                         "differences (no-op while VALIDATE_ALLOW_LIST is "
                         "empty)")
    args = ap.parse_args(argv)

    if args.validate:
        md_result = run_extractor(
            "markdown",
            md_path=args.input,
            compile_db=args.compile_db,
            sim_root=args.sim_root,
        )
        ast_result = run_extractor(
            "ast",
            md_path=args.input,
            compile_db=args.compile_db,
            sim_root=args.sim_root,
        )
        if md_result.errors or ast_result.errors:
            print_extraction_errors("markdown", md_result)
            print_extraction_errors("ast", ast_result)
            return 2
        had_diff, lines = diff_results(ast_result, md_result, strict=args.strict)
        for line in lines:
            print(line)
        if not had_diff:
            print("validate: no unexpected differences")
        return 1 if had_diff else 0

    result = run_extractor(
        args.source,
        md_path=args.input,
        compile_db=args.compile_db,
        sim_root=args.sim_root,
    )
    if result.errors:
        print_extraction_errors(args.source, result)
        return 2

    def _is_summary(w: str) -> bool:
        # Each extractor's leading warning is an informational summary
        # ("parsed N rows..." for markdown, "AST extractor: parsed N
        # TUs..." for AST). They are not findings.
        return w.startswith("parsed ") or w.startswith("AST extractor:")

    for w in result.warnings:
        if _is_summary(w):
            print(w)
        else:
            print(f"warning: {w}", file=sys.stderr)

    if not result.modules and not result.edges:
        print("error: extractor produced no output", file=sys.stderr)
        return 2

    if args.check:
        findings = [w for w in result.warnings if not _is_summary(w)]
        return 1 if findings else 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dot_path = args.out_dir / "signal_diagram.dot"
    mmd_path = args.out_dir / "signal_diagram.mmd"
    drawio_path = args.out_dir / "signal_diagram.drawio"
    dot_path.write_text(emit_dot(result.modules, result.edges, source=args.source))
    mmd_path.write_text(emit_mermaid(result.modules, result.edges))
    drawio_path.write_text(emit_drawio(result.modules, result.edges, source=args.source))
    print(f"wrote {dot_path}")
    print(f"wrote {mmd_path}")
    print(f"wrote {drawio_path}")

    if args.svg:
        # The poster uses pinned `pos` coordinates, so it must be laid
        # out by neato (the `layout=neato` attribute in the .dot would
        # also redirect a bare `dot`, but invoking neato directly is
        # explicit and avoids depending on that attribute).
        if shutil.which("neato") is None:
            print("warning: `neato` not found on PATH; skipping SVG/PNG render",
                  file=sys.stderr)
        else:
            for fmt in ("svg", "png"):
                out_path = args.out_dir / f"signal_diagram.{fmt}"
                # neato may emit non-fatal warnings and exit nonzero even
                # though it wrote the file. Treat the file's existence
                # and non-zero size as the success signal.
                render = subprocess.run(
                    ["neato", f"-T{fmt}", str(dot_path), "-o", str(out_path)],
                    capture_output=True, text=True,
                )
                if out_path.exists() and out_path.stat().st_size > 0:
                    print(f"wrote {out_path}")
                    if render.returncode != 0:
                        print(f"note: neato reported a warning (exit "
                              f"{render.returncode}): {render.stderr.strip()}",
                              file=sys.stderr)
                else:
                    print(f"error: neato failed (exit {render.returncode}): "
                          f"{render.stderr}", file=sys.stderr)
                    return render.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
