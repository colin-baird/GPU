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
`diagram_types.py`); this script emits Graphviz DOT, Mermaid, and
optionally SVG from those records.

Output files (under `tools/` by default):
  - `signal_diagram.dot` — Graphviz source.
  - `signal_diagram.mmd` — Mermaid companion for inline embedding.
  - `signal_diagram.svg` — only when `--svg` is passed and `dot` is on PATH.

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
    """Emit a Graphviz DOT representation of the diagram."""
    out: list[str] = []
    out.append("digraph signal_flow {")
    out.append('  rankdir=TB;')
    # Title and a small inline legend live in an HTML graph label so the
    # legend doesn't form a disconnected component (which used to confuse
    # graphviz's rank assignment).
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

    cluster_of: dict[str, str] = {m.name: m.cluster for m in modules}
    by_cluster: dict[str, list[str]] = defaultdict(list)
    for m in modules:
        by_cluster[m.cluster].append(m.name)

    for cluster in CLUSTER_ORDER:
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
        # constraint=false marks edges that shouldn't influence rank
        # assignment, so layout follows the main data-flow DAG and these
        # edges float around it.
        # - back-pressure: flow backward against data direction.
        # - Control-cluster edges (PanicController, TimingModel): a
        #   side-channel observer; letting these edges constrain ranks
        #   pulls TimingModel away from PanicController and stretches
        #   the Control cluster across the whole graph height.
        if (back_pressure
                or cluster_of.get(src) == "Control"
                or cluster_of.get(dst) == "Control"):
            attrs.append("constraint=false")
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
                    help="Also render an SVG via `dot` if available")
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
    dot_path.write_text(emit_dot(result.modules, result.edges, source=args.source))
    mmd_path.write_text(emit_mermaid(result.modules, result.edges))
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
            svg_result = subprocess.run(
                ["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)],
                capture_output=True, text=True,
            )
            if svg_path.exists() and svg_path.stat().st_size > 0:
                print(f"wrote {svg_path}")
                if svg_result.returncode != 0:
                    print(f"note: dot reported a warning (exit {svg_result.returncode}): {svg_result.stderr.strip()}", file=sys.stderr)
            else:
                print(f"error: dot failed (exit {svg_result.returncode}): {svg_result.stderr}", file=sys.stderr)
                return svg_result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
