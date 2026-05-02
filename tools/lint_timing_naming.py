#!/usr/bin/env python3
"""Lint cross-stage accessor naming in the timing model.

Scans every header under `sim/include/gpu_sim/timing/` and reports
violations of the rules documented in
`/resources/cpp_coding_standard.md` § Cross-stage accessor naming and
`/resources/timing_discipline.md`.

Four checks run in order:

  *Prefix layer* — every public const member function returning `bool`,
  `std::optional<…>`, a payload struct, or a payload reference must
  start with `current_` or `next_`. Lifecycle hooks (`evaluate`,
  `commit`, `reset`, `flush`, `seed_next`, `accept`, `consume_result`,
  `add_source`, etc.) are exempt via the explicit `LIFECYCLE_METHODS`
  allowlist.

  *Postfix layer* — for each accessor, classify by return type and name
  shape:
    - `bool` and matches `^(current|next)_has_<noun>` → possession
      predicate, OK.
    - `bool` otherwise → expected state predicate; flag any `_is_*` /
      `_has_*` mid-name patterns that suggest the wrong shape.
    - non-`bool` → expected payload accessor; flag adjective tells
      (`_busy`, `_idle`, `_stalled`, `_full`, `_empty`).

  *Polarity layer* — flag pairs of accessors on the same class whose
  names suggest inverse meanings (`busy`/`ready`, `stalled`/`advancing`,
  `full`/`empty` etc.). Only one polarity per concept.

  *Field-shape layer* — fields named `current_*` / `next_*` must be
  declared in a `private:` section. Public bare references to other
  module classes held as members are flagged as candidates for the
  post-wired-pointer rule (human decides).

Initial mode is report-only (Phase 0): the script prints findings to
stdout but exits 0. Phase 6 flips the default to enforcement; the
`--report-only` flag preserves the legacy behavior for local
exploration. CI invokes the script without flags.

Per-line annotation `// timing-naming-allow: <reason>` silences a
single finding on the annotated line.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HEADER_DIR = REPO_ROOT / "sim" / "include" / "gpu_sim" / "timing"

# Classes the lint inspects. Other classes/structs declared in the
# timing/ headers (data POD structs, helper buffers like MSHRFile and
# InstructionBuffer, branch predictor) are out of scope for the
# cross-stage accessor convention. Listed explicitly so a new module
# entering the diagram has to be added here too — keeps the lint and
# the diagram extractor in sync.
TIMING_MODULES: set[str] = {
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
    "FixedLatencyMemory",
    "DRAMSim3Memory",
    "WritebackArbiter",
    "PanicController",
    "TimingModel",
    "PipelineStage",
    "ExecutionUnit",
    "QueuedWritebackSource",
}

# Methods exempt from the prefix rule. Each entry below was verified to
# suppress at least one finding on a TIMING_MODULES class — non-const
# lifecycle methods (evaluate / commit / reset / flush / accept etc.)
# don't need entries here because the prefix layer only inspects public
# const accessors, so non-const methods are skipped by construction.
# Same for private helpers (the access-tracker filters them out).
#
# Audit checklist when adding a new entry:
#   (1) Is the accessor read by another module's evaluate() / commit()
#       in a flow-control role? If yes, it IS a cross-stage signal —
#       rename it with the current_*/next_* prefix instead of allow-
#       listing.
#   (2) Does the accessor's underlying field follow the discipline
#       (`current_*` / `next_*` double-buffered)? If yes, the natural
#       name is the prefixed form; allow-listing is hiding a real miss.
#   (3) If neither — the accessor is genuinely scalar telemetry,
#       configuration, an orchestrator-status latch, or a snapshot view
#       — the entry belongs here.
LIFECYCLE_METHODS: set[str] = {
    # ───── Stat counters and capacity reporters ─────────────────────
    # These return scalars derived from internal subsystem state
    # (MSHR active count, queue depths, write-buffer occupancy,
    # configuration-driven sizes). They are consumed by tests and the
    # trace exporter, never by another module's flow-control logic.
    "active_mshr_count",
    "active_mshr_warps",
    "secondary_mshr_count",
    "pinned_line_count",
    "write_buffer_size",
    "in_flight_count",
    "response_count",
    "request_fifo_size",
    "max_observed_response_queue",
    "response_queue_capacity",
    "dram_ticks",
    "chunks_per_line",
    "ready_source_count",
    "pipeline_occupancy",
    "queue_depth",
    "num_buffers",
    "cycle_count",

    # ───── Snapshot views ───────────────────────────────────────────
    # Vector / reference returns derived from committed state, used by
    # tests and the trace exporter only.
    "active_warps",
    "pipeline_snapshot",
    "mshrs",
    "buffer",

    # ───── Orchestrator state-machine status ────────────────────────
    # PanicController is a one-shot state machine — these fields are
    # set once by `trigger()` / mutated by `evaluate()` step transitions
    # and observed by TimingModel::tick(). They are not double-buffered
    # cross-stage signals; the prefix rule's premise (next_*/current_*
    # field pair) doesn't apply.
    "is_active",
    "is_done",
    "step",
    "panic_warp",
    "panic_pc",
    "panic_cause",

    # ───── Drain / aggregate-idle predicates ────────────────────────
    # Distinct from the back-pressure `current_busy()` signal: an idle
    # subsystem is fully drained (no work in any stage), whereas a
    # busy/stalled module is rejecting *new* work this cycle. The two
    # are not inverses, so allow-listing the drain check does not
    # collide with the polarity rule.
    "is_idle",
    "is_coalesced",  # CoalescingUnit per-cycle FSM state, post-commit query

    # ───── Per-unit snapshot accessors ──────────────────────────────
    # Each unit exposes `busy()` / `active_warp()` / `pending_input()`
    # / `pending_entry()` / `result_entry()` for `build_cycle_snapshot`
    # and the structured trace. These read committed (`current_*`)
    # state, but a cross-stage flow-control consumer reads the
    # `current_busy()` back-pressure signal instead — so these names
    # don't clash with the discipline at any cross-stage call site.
    "busy",
    "active_warp",
    "resident_warp",
    "pending_input",
    "pending_entry",
    "result_entry",
    "serial_index",         # CoalescingUnit per-cycle counter snapshot
}

# Words that indicate a state predicate when seen as a standalone or
# trailing token — used by the postfix layer to flag adjectives in
# payload-accessor names.
ADJECTIVE_TELLS: set[str] = {
    "busy",
    "idle",
    "stalled",
    "full",
    "empty",
    "ready",
    "active",
    "pending",
    "claimed",
}

# Inverse-polarity word pairs flagged by the polarity layer.
INVERSE_PAIRS: list[tuple[str, str]] = [
    ("busy", "ready"),
    ("busy", "free"),
    ("busy", "idle"),
    ("stalled", "advancing"),
    ("stalled", "ready"),
    ("full", "empty"),
    ("active", "idle"),
    ("pending", "ready"),
]

ALLOW_ANNOTATION_RE = re.compile(r"//\s*timing-naming-allow:")


@dataclass
class Finding:
    file: Path
    line: int
    rule: str
    message: str

    def format(self, root: Path) -> str:
        try:
            rel = self.file.relative_to(root)
        except ValueError:
            rel = self.file
        return f"{rel}:{self.line}: [{self.rule}] {self.message}"


@dataclass
class ClassInfo:
    name: str
    start_line: int
    methods: list[tuple[str, int, str, str]] = field(default_factory=list)
    # (name, line, return_type, access)
    fields: list[tuple[str, int, str, str]] = field(default_factory=list)
    # (name, line, type, access)


# Regex to capture a public const accessor's signature. Captures:
#   1: return type (rough, may include `const`, `&`, `*`, template args)
#   2: method name
PUBLIC_CONST_METHOD_RE = re.compile(
    r"""^\s*
        (?:virtual\s+)?
        (?:static\s+)?
        (?:inline\s+)?
        (?P<ret>
            (?:const\s+)?
            [A-Za-z_:][\w:<>,\s\*&]*?
        )
        \s+
        (?P<name>[A-Za-z_]\w*)
        \s*\(
        (?P<args>[^)]*)
        \)
        \s*const
        \s*(?:override)?
        \s*(?:=\s*(?:0|default))?
        \s*[;{]
    """,
    re.VERBOSE,
)

# Regex to capture a member field declaration. Captures:
#   1: type
#   2: name
FIELD_RE = re.compile(
    r"""^\s*
        (?P<type>
            (?:const\s+|static\s+|mutable\s+)*
            [A-Za-z_:][\w:<>,\s\*&]*?
        )
        \s+
        (?P<name>[A-Za-z_]\w*)
        \s*(?:=\s*[^;]+)?
        \s*[;{]
    """,
    re.VERBOSE,
)

# Class/struct opener.
CLASS_OPEN_RE = re.compile(r"^\s*(?:class|struct)\s+([A-Za-z_]\w*)\b")


def _strip_comments(text: str) -> str:
    """Strip // and /* */ comments to reduce false matches in regex."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def _parse_header(path: Path) -> list[ClassInfo]:
    """Parse a header into ClassInfo records.

    Light-touch: tracks brace depth to find class boundaries and
    public/private/protected sections. Skips non-module classes.
    """
    classes: list[ClassInfo] = []
    text = path.read_text()
    lines = text.splitlines()

    # Map line → access section by walking with a brace counter.
    cls: ClassInfo | None = None
    cls_brace_open: int | None = None
    brace_depth = 0
    access = "private"  # default for class; we'll override on entry

    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        # Skip preprocessor and pure comment lines for class detection.
        compact = _strip_comments(raw)

        # Track class entry.
        if cls is None:
            m = CLASS_OPEN_RE.match(compact)
            if m and m.group(1) in TIMING_MODULES:
                cls = ClassInfo(name=m.group(1), start_line=i + 1)
                cls_brace_open = None
                # Default access depends on keyword.
                kw = compact.split()[0]
                access = "public" if kw == "struct" else "private"
        # Track brace depth & access transitions inside an open class.
        if cls is not None:
            opens = compact.count("{")
            closes = compact.count("}")
            if cls_brace_open is None and opens > 0:
                cls_brace_open = brace_depth + 1
            brace_depth_before = brace_depth
            brace_depth += opens - closes

            if cls_brace_open is not None:
                # access transition lines like "public:" / "private:" /
                # "protected:" that appear at the class scope (depth ==
                # cls_brace_open).
                if (brace_depth == cls_brace_open
                        and brace_depth_before == cls_brace_open):
                    label_match = re.match(
                        r"^\s*(public|private|protected)\s*:", compact)
                    if label_match:
                        access = label_match.group(1)

                # Member detection: any line that *starts* at class scope
                # (brace_depth_before == cls_brace_open) is a candidate
                # member declaration. We match regardless of whether the
                # line opens a brace — multi-line inline method
                # definitions ("ReturnT name() const { ... }" split
                # across several lines) start at class scope and open a
                # brace, but the signature is fully on the opening line.
                if brace_depth_before == cls_brace_open:
                    method_match = PUBLIC_CONST_METHOD_RE.match(compact)
                    if method_match and access == "public":
                        ret = method_match.group("ret").strip()
                        name = method_match.group("name")
                        cls.methods.append(
                            (name, i + 1, ret, access)
                        )
                    elif brace_depth == cls_brace_open:
                        # Field declaration heuristic: only recognize
                        # lines that don't change brace depth (so we
                        # don't mistake a method body opener for a
                        # field), end with `;`, and aren't
                        # function-shaped (no `(` outside template
                        # args).
                        if (";" in compact and "(" not in compact
                                and "operator" not in compact):
                            fm = FIELD_RE.match(compact)
                            if fm:
                                name = fm.group("name")
                                type_ = fm.group("type").strip()
                                if name not in ("public", "private", "protected"):
                                    cls.fields.append(
                                        (name, i + 1, type_, access)
                                    )

                # Class close.
                if brace_depth < cls_brace_open:
                    classes.append(cls)
                    cls = None
                    cls_brace_open = None
                    access = "private"
        i += 1
    return classes


def _is_payload_return(ret: str) -> bool:
    """Return True if the return type is non-bool (a payload type)."""
    ret_clean = ret.replace("override", "").strip()
    # Strip leading const/virtual/static.
    return not re.search(r"\bbool\b\s*$", ret_clean) and not ret_clean.endswith("bool")


def _allowed_on_line(line_text: str) -> bool:
    return bool(ALLOW_ANNOTATION_RE.search(line_text))


def _check_prefix(cls: ClassInfo, header: list[str], path: Path,
                  findings: list[Finding]) -> None:
    """Prefix layer: public const accessors must start with current_ or next_."""
    for (name, line, ret, access) in cls.methods:
        if access != "public":
            continue
        if name in LIFECYCLE_METHODS:
            continue
        if name.startswith("set_") or name.startswith("get_"):
            continue
        if _allowed_on_line(header[line - 1]):
            continue
        if not (name.startswith("current_") or name.startswith("next_")):
            findings.append(Finding(
                file=path, line=line, rule="prefix",
                message=(
                    f"{cls.name}::{name}() — public const accessor missing "
                    f"current_/next_ prefix; cycle discipline must be "
                    f"encoded in the name (REGISTERED → current_*, "
                    f"COMBINATIONAL → next_*)"
                ),
            ))


def _check_postfix(cls: ClassInfo, header: list[str], path: Path,
                   findings: list[Finding]) -> None:
    """Postfix layer: state predicates avoid is_/has_ filler; payload
    accessors avoid adjective tells."""
    for (name, line, ret, access) in cls.methods:
        if access != "public":
            continue
        if name in LIFECYCLE_METHODS:
            continue
        if not (name.startswith("current_") or name.startswith("next_")):
            continue
        if _allowed_on_line(header[line - 1]):
            continue
        is_payload = _is_payload_return(ret)
        body = name.split("_", 1)[1] if "_" in name else ""
        # Possession predicate?
        if body.startswith("has_"):
            if is_payload:
                findings.append(Finding(
                    file=path, line=line, rule="postfix",
                    message=(
                        f"{cls.name}::{name}() — has_<noun> form must "
                        f"return bool, but returns '{ret}'"
                    ),
                ))
            continue
        # Bool state predicate.
        if not is_payload:
            if "is_" in body:
                findings.append(Finding(
                    file=path, line=line, rule="postfix",
                    message=(
                        f"{cls.name}::{name}() — state predicate must "
                        f"use bare adjective (drop is_ filler)"
                    ),
                ))
            continue
        # Payload accessor.
        # Flag adjective tells in trailing word.
        tail = body.rsplit("_", 1)[-1] if body else ""
        if tail in ADJECTIVE_TELLS:
            findings.append(Finding(
                file=path, line=line, rule="postfix",
                message=(
                    f"{cls.name}::{name}() — payload accessor name ends "
                    f"in adjective '{tail}'; payload accessors should "
                    f"use a bare noun"
                ),
            ))


def _check_polarity(cls: ClassInfo, header: list[str], path: Path,
                    findings: list[Finding]) -> None:
    """Polarity layer: flag inverse pairs on the same class."""
    method_names = {name for (name, _line, _ret, access) in cls.methods
                    if access == "public"}
    method_lines = {name: line for (name, line, _ret, access) in cls.methods
                    if access == "public"}
    for (a, b) in INVERSE_PAIRS:
        for prefix in ("current", "next", ""):
            if prefix:
                full_a = f"{prefix}_{a}"
                full_b = f"{prefix}_{b}"
            else:
                full_a = a
                full_b = b
            if full_a in method_names and full_b in method_names:
                line_a = method_lines.get(full_a, 0)
                if _allowed_on_line(header[line_a - 1] if line_a else ""):
                    continue
                findings.append(Finding(
                    file=path, line=line_a or cls.start_line,
                    rule="polarity",
                    message=(
                        f"{cls.name} exposes inverse-polarity pair "
                        f"{full_a}() and {full_b}(); pick the "
                        f"asserted=blocking polarity and drop the other"
                    ),
                ))


def _check_field_shape(cls: ClassInfo, header: list[str], path: Path,
                       findings: list[Finding]) -> None:
    """Field-shape layer: current_*/next_* fields must be private."""
    for (name, line, type_, access) in cls.fields:
        if not (name.startswith("current_") or name.startswith("next_")):
            continue
        if _allowed_on_line(header[line - 1]):
            continue
        if access != "private":
            findings.append(Finding(
                file=path, line=line, rule="field-shape",
                message=(
                    f"{cls.name}::{name} — REGISTERED state field is "
                    f"'{access}' but must be private; expose via "
                    f"current_*()/next_*() accessor"
                ),
            ))


def lint_header(path: Path) -> list[Finding]:
    """Lint a single header file and return findings."""
    findings: list[Finding] = []
    text = path.read_text()
    lines = text.splitlines()
    classes = _parse_header(path)
    for cls in classes:
        _check_prefix(cls, lines, path, findings)
        _check_postfix(cls, lines, path, findings)
        _check_polarity(cls, lines, path, findings)
        _check_field_shape(cls, lines, path, findings)
    return findings


def lint_directory(header_dir: Path) -> list[Finding]:
    """Lint every .h file under `header_dir` and return findings sorted
    by (file, line)."""
    findings: list[Finding] = []
    for path in sorted(header_dir.rglob("*.h")):
        findings.extend(lint_header(path))
    findings.sort(key=lambda f: (str(f.file), f.line, f.rule))
    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--header-dir", type=Path, default=DEFAULT_HEADER_DIR,
        help=("Directory of timing headers to scan (default: "
              "%(default)s)"),
    )
    ap.add_argument(
        "--report-only", action="store_true",
        help=("Always exit 0 regardless of findings. Useful for local "
              "exploration; CI invokes the script without this flag and "
              "treats any finding as a failure (Phase 6 enforcement)."),
    )
    args = ap.parse_args(argv)

    if not args.header_dir.is_dir():
        print(f"error: header directory not found: {args.header_dir}",
              file=sys.stderr)
        return 2

    findings = lint_directory(args.header_dir)
    for f in findings:
        print(f.format(REPO_ROOT))

    if not findings:
        print("lint_timing_naming: clean")
        return 0

    if args.report_only:
        print(
            f"\nlint_timing_naming: {len(findings)} finding(s) "
            "(report-only mode)",
            file=sys.stderr,
        )
        return 0

    print(
        f"\nlint_timing_naming: {len(findings)} finding(s) — failing "
        "build. Suppress a single finding with a per-line comment like "
        "`// timing-naming-allow: <reason>`; if the rule needs broader "
        "tweaking, edit tools/lint_timing_naming.py.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
