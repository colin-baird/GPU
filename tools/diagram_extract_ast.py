"""libclang-based AST extractor for the signal-flow diagram.

Walks the timing-model translation units listed in
`build/compile_commands.json`. The C++ source is the source of truth;
this extractor populates `Module` and `Edge` records that the renderer
consumes. The markdown extractor under `diagram_extract_md.py` remains
available for `--source=markdown` and `--validate` cross-checks.

Each TU is parsed once and indexed in a single project-scoped pass
(`_build_index`): the walk descends only into top-level cursors located
under `sim/`, skipping the STL/system-header subtrees entirely, and
records three maps consulted by every later pass —

  - `decl_map`     — class name -> class-definition cursor
  - `method_index` — class name -> {method name -> definition cursor}
  - `timing_ctor`  — the `TimingModel` constructor definition

Passes B/C then look up the index instead of re-walking every TU. This
keeps the cost bounded by the project's own AST rather than the full
preprocessed TU; module classes, their methods, and the `TimingModel`
constructor all live under `sim/`, so nothing the extractor needs is
pruned away and the output is unchanged.

Pass overview (numbered to match `project-plans/snazzy-sprouting-charm.md`):

  - **Pass A — Module discovery.** Classifies every indexed
    CXXRecordDecl. A class is a module if it transitively inherits
    `PipelineStage` or `ExecutionUnit`, or appears in
    `STANDALONE_MODULES`.
  - **Pass B — Wiring discovery.** Walks `TimingModel`'s constructor
    body for `add_source` and `set_*` calls plus the lambda-wired
    `panic_->set_drained_query(...)` case. Builds a map from each
    module's member fields to the concrete module class on the other
    end.
  - **Pass C — Edge discovery.** For every module's `evaluate()` and
    `commit()` body, collects cross-module `member_->method()` calls and
    `member_.field` reads. Same-class private helpers are inlined to
    depth 2 so e.g. `CoalescingUnit`'s helpers attribute to the right
    edge.
  - **Pass D — Classification.** Under the two-axis model, the cycle
    axis is REGISTERED (default) or COMBINATIONAL (callee name starts
    with `next_`, or carved out by `EDGE_CLASSIFICATION_OVERRIDES`).
    Back-pressure direction is derived by the renderer from cluster
    topology (`MODULE_ORDER`) and is not part of the classification.

Importing this module fails with a clear error if the `clang` Python
bindings are not installed; install via:

    pip install -r tools/requirements.txt

or, on Debian/Ubuntu, `apt install python3-clang`.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from collections import defaultdict
from pathlib import Path
from typing import Iterable

try:
    import clang.cindex as cindex
except ImportError as exc:
    raise ImportError(
        "AST extractor requires the libclang Python bindings. Install via "
        "`pip install -r tools/requirements.txt` or `apt install python3-clang`."
    ) from exc

from diagram_types import Edge, ExtractionResult, Module
from pipeline_order import MODULE_ORDER as _PIPELINE_ORDER

# Cluster assignment for every module the extractor may discover. Mirrors
# the markdown extractor's MODULES list so module sets can be diffed.
MODULE_CLUSTER: dict[str, str] = {
    "FetchStage":              "Frontend & Issue",
    "DecodeStage":             "Frontend & Issue",
    "WarpScheduler":           "Frontend & Issue",
    "Scoreboard":              "Frontend & Issue",
    "BranchShadowTracker":     "Frontend & Issue",
    "OperandCollector":        "Frontend & Issue",
    "ALUUnit":                 "Execute",
    "MultiplyUnit":            "Execute",
    "DivideUnit":              "Execute",
    "TLookupUnit":             "Execute",
    "LdStUnit":                "Execute",
    "CoalescingUnit":          "Memory",
    "LoadGatherBufferFile":    "Memory",
    "L1Cache":                 "Memory",
    "ExternalMemoryInterface": "Memory",
    "WritebackArbiter":        "Writeback",
    "PanicController":         "Control",
    "TimingModel":             "Control",
}

# Render order within each cluster. Sourced from the shared
# `pipeline_order` module so the AST extractor, the markdown extractor,
# the renderer, and `lint_timing_naming.py` all share one ordering.
MODULE_ORDER: list[str] = list(_PIPELINE_ORDER)

# Modules that don't inherit PipelineStage or ExecutionUnit. They still
# appear as nodes in the diagram because cross-stage edges land on them.
STANDALONE_MODULES: set[str] = {
    "TimingModel",
    "Scoreboard",
    "BranchShadowTracker",
    "CoalescingUnit",
    "L1Cache",
    "ExternalMemoryInterface",
    "PanicController",
}

# Names of base classes whose transitive subclasses are diagram modules.
HIERARCHY_BASES: set[str] = {"PipelineStage", "ExecutionUnit"}

# Intermediate base classes that inherit a HIERARCHY_BASE but are not
# themselves drawable modules.
EXCLUDED_FROM_HIERARCHY: set[str] = {"QueuedWritebackSource"}

# `L1Cache` is the diagram name; the C++ class is also `L1Cache`.
# `ExternalMemoryInterface` is an abstract interface; its concrete
# subclasses (FixedLatencyMemory, DRAMSim3Memory) are mapped to the
# interface name for diagram purposes.
TYPE_REWRITE: dict[str, str] = {
    "FixedLatencyMemory":      "ExternalMemoryInterface",
    "DRAMSim3Memory":          "ExternalMemoryInterface",
    "QueuedWritebackSource":   "LoadGatherBufferFile",
}


def _load_compile_db(compile_db: Path) -> list[dict]:
    """Parse compile_commands.json. Errors out with an actionable message
    if the file is missing or empty."""
    if not compile_db.exists():
        raise FileNotFoundError(
            f"compile_commands.json not found at {compile_db}.\n"
            f"  Generate it with: cmake -B build && cmake --build build -j8"
        )
    entries = json.loads(compile_db.read_text())
    if not entries:
        raise ValueError(
            f"compile_commands.json at {compile_db} is empty; rerun cmake."
        )
    return entries


def _select_timing_tus(entries: list[dict], sim_root: Path) -> list[dict]:
    """Filter the compile DB to translation units under sim/src/timing/."""
    timing_dir = (sim_root / "src" / "timing").resolve()
    selected: list[dict] = []
    for entry in entries:
        file = entry.get("file", "")
        directory = entry.get("directory", "")
        full = Path(file) if os.path.isabs(file) else Path(directory) / file
        try:
            full = full.resolve()
        except OSError:
            continue
        try:
            full.relative_to(timing_dir)
        except ValueError:
            continue
        selected.append(entry)
    return selected


def _split_command(entry: dict) -> list[str]:
    """Extract the compiler argument list from a compile_commands entry,
    dropping the leading compiler executable token and the trailing input
    file."""
    if "arguments" in entry:
        argv = list(entry["arguments"])
    else:
        argv = shlex.split(entry["command"])
    if argv:
        argv = argv[1:]  # drop compiler executable
    file_arg = entry["file"]
    if argv and (argv[-1] == file_arg or argv[-1].endswith(file_arg)):
        argv = argv[:-1]
    return [a for a in argv if a not in ("-c",)]


def _direct_bases(cursor: cindex.Cursor) -> list[str]:
    """Return the spelling of every direct base class of a record decl."""
    bases: list[str] = []
    for child in cursor.get_children():
        if child.kind == cindex.CursorKind.CXX_BASE_SPECIFIER:
            referenced = child.referenced
            if referenced is not None:
                bases.append(referenced.spelling)
            else:
                bases.append(child.type.spelling.split("::")[-1])
    return bases


def _transitive_bases(name: str, base_map: dict[str, list[str]],
                      seen: set[str] | None = None) -> set[str]:
    """All transitive base-class names for `name`, walking `base_map`."""
    if seen is None:
        seen = set()
    out: set[str] = set()
    for b in base_map.get(name, []):
        if b in seen:
            continue
        seen.add(b)
        out.add(b)
        out |= _transitive_bases(b, base_map, seen)
    return out


# ----------------------------------------------------------------------------
# Type unwrapping
# ----------------------------------------------------------------------------

def _unwrap_type_name(t: cindex.Type) -> str | None:
    """Reduce a C++ type to a bare class spelling.

    Strips pointer / reference / const / unique_ptr<T> wrappers and
    returns the unqualified class name, or `None` if it doesn't resolve
    to a named record. `TYPE_REWRITE` is applied to collapse abstract
    interfaces and intermediate bases to their diagram-facing name.
    """
    if t is None:
        return None
    canonical = t.get_canonical()
    # Strip pointer / reference layers.
    while canonical.kind in (cindex.TypeKind.POINTER, cindex.TypeKind.LVALUEREFERENCE,
                             cindex.TypeKind.RVALUEREFERENCE):
        canonical = canonical.get_pointee().get_canonical()

    decl = canonical.get_declaration()
    name = decl.spelling if decl is not None and decl.spelling else None
    if not name:
        # Fall back to parsing the textual spelling.
        spelling = t.spelling
        spelling = re.sub(r"\bconst\b", "", spelling).strip()
        spelling = spelling.rstrip("*&").strip()
        spelling = spelling.split("::")[-1]
        if "<" in spelling:
            spelling = spelling.split("<", 1)[0]
        name = spelling or None

    if name is None:
        return None

    # std::unique_ptr<T> / std::shared_ptr<T>: unwrap to T.
    if name in ("unique_ptr", "shared_ptr"):
        # Look at the first template argument.
        for arg_idx in range(canonical.get_num_template_arguments()):
            arg = canonical.get_template_argument_type(arg_idx)
            inner = _unwrap_type_name(arg)
            if inner:
                return inner
        return None

    return TYPE_REWRITE.get(name, name)


# ----------------------------------------------------------------------------
# Single-pass project-scoped indexing
# ----------------------------------------------------------------------------

def _build_index(translation_units: list[cindex.TranslationUnit],
                 sim_root: Path
                 ) -> tuple[set[str], dict[str, list[str]],
                            dict[str, cindex.Cursor],
                            dict[str, dict[str, cindex.Cursor]],
                            cindex.Cursor | None]:
    """Walk every TU once, descending only into project (`sim/`) cursors.

    Returns `(seen_classes, base_map, decl_map, method_index,
    timing_ctor)`:

      - `seen_classes`  — every project class/struct spelling defined.
      - `base_map`      — class spelling -> list of direct base spellings
        (accumulated across all definitions seen; `_transitive_bases`
        dedups, so duplicates are harmless).
      - `decl_map`      — class spelling -> first class-definition cursor.
      - `method_index`  — class spelling -> {method spelling -> definition
        cursor}. Covers both in-class inline definitions and out-of-line
        definitions (both carry `semantic_parent == class`).
      - `timing_ctor`   — the first `TimingModel` constructor definition.

    The traversal cost is bounded by the project's own AST: each TU's
    top-level cursors are filtered by source location, so the STL and
    other system-header subtrees are never descended into. Module
    classes, their methods, and the `TimingModel` constructor all live
    under `sim/`, so nothing the extractor needs is pruned away.
    """
    sim_root_str = str(sim_root.resolve())
    sim_prefix = sim_root_str + os.sep

    Kind = cindex.CursorKind
    record_kinds = {Kind.CLASS_DECL, Kind.STRUCT_DECL, Kind.CLASS_TEMPLATE}

    seen_classes: set[str] = set()
    base_map: dict[str, list[str]] = {}
    decl_map: dict[str, cindex.Cursor] = {}
    method_index: dict[str, dict[str, cindex.Cursor]] = defaultdict(dict)
    timing_ctor: cindex.Cursor | None = None

    # Cache the under-sim/ verdict per source-file path string. A TU has
    # one top-level cursor per declaration from every included header, so
    # the same header path recurs thousands of times across 20 TUs.
    under_cache: dict[str, bool] = {}

    def _under_sim(file_obj: object) -> bool:
        if file_obj is None:
            return False
        name = file_obj.name
        verdict = under_cache.get(name)
        if verdict is None:
            verdict = os.path.abspath(name).startswith(sim_prefix)
            under_cache[name] = verdict
        return verdict

    for tu in translation_units:
        # The TU cursor's direct children are the flattened top-level
        # declarations from the main file and every included header. A
        # system-header `namespace std { ... }` block is a single such
        # child whose entire subtree we skip by not descending into it.
        for top in tu.cursor.get_children():
            if not _under_sim(top.location.file):
                continue
            for cursor in top.walk_preorder():
                kind = cursor.kind
                if kind in record_kinds:
                    if not cursor.is_definition():
                        continue
                    spelling = cursor.spelling
                    if not spelling:
                        continue
                    seen_classes.add(spelling)
                    base_map.setdefault(spelling, []).extend(
                        _direct_bases(cursor))
                    decl_map.setdefault(spelling, cursor)
                elif kind == Kind.CXX_METHOD:
                    if not cursor.is_definition():
                        continue
                    parent = cursor.semantic_parent
                    if parent is not None and parent.spelling:
                        method_index[parent.spelling][cursor.spelling] = cursor
                elif kind == Kind.CONSTRUCTOR:
                    if timing_ctor is not None or not cursor.is_definition():
                        continue
                    parent = cursor.semantic_parent
                    if parent is not None and parent.spelling == "TimingModel":
                        timing_ctor = cursor

    return seen_classes, base_map, decl_map, dict(method_index), timing_ctor


# ----------------------------------------------------------------------------
# Pass A — module discovery
# ----------------------------------------------------------------------------

def _discover_modules(seen_classes: set[str],
                      base_map: dict[str, list[str]]
                      ) -> tuple[list[Module], list[str]]:
    """Pass A — module discovery.

    Classifies the indexed classes. A class is a module if it
    transitively inherits a `HIERARCHY_BASE`, or is a `STANDALONE_MODULE`
    that was seen. Returns `(ordered_modules, warnings)`.
    """
    discovered: set[str] = set()
    for name in seen_classes:
        if name in EXCLUDED_FROM_HIERARCHY:
            continue
        if _transitive_bases(name, base_map) & HIERARCHY_BASES:
            discovered.add(name)

    warnings: list[str] = []
    for name in STANDALONE_MODULES:
        if name not in seen_classes:
            warnings.append(
                f"standalone module {name!r} not found in any timing TU"
            )
    discovered |= (STANDALONE_MODULES & seen_classes)

    expected = set(MODULE_CLUSTER)
    missing = expected - discovered
    if missing:
        warnings.append(
            f"expected modules not discovered in AST: {sorted(missing)}"
        )
    extra = discovered - expected
    if extra:
        warnings.append(
            f"discovered modules without a cluster assignment: {sorted(extra)}"
        )

    ordered: list[Module] = []
    for name in MODULE_ORDER:
        if name in discovered:
            ordered.append(Module(name=name, cluster=MODULE_CLUSTER[name]))
    for name in sorted(discovered - set(MODULE_ORDER)):
        ordered.append(Module(name=name, cluster="Other"))
    return ordered, warnings


# ----------------------------------------------------------------------------
# Field-type maps
# ----------------------------------------------------------------------------

def _build_field_type_map(decl_map: dict[str, cindex.Cursor],
                          modules: set[str]) -> dict[str, dict[str, str]]:
    """For each module, map field-name -> module-class-name.

    Includes pointer/reference/unique_ptr fields whose unwrapped type
    resolves to another module. Field names are stored verbatim
    (including trailing underscore where present).
    """
    out: dict[str, dict[str, str]] = {}
    for name, cursor in decl_map.items():
        if name not in modules:
            continue
        fields: dict[str, str] = {}
        for child in cursor.get_children():
            if child.kind != cindex.CursorKind.FIELD_DECL:
                continue
            target = _unwrap_type_name(child.type)
            if target and target in modules and target != name:
                fields[child.spelling] = target
        out[name] = fields
    return out


# ----------------------------------------------------------------------------
# Pass B — wiring discovery
# ----------------------------------------------------------------------------

def _resolve_arg_module(arg: cindex.Cursor, ctor_field_map: dict[str, str],
                        modules: set[str]) -> str | None:
    """Return the module name that an argument expression resolves to.

    Handles `xxx_.get()`, `&xxx_`, `*xxx_`, parens, and direct field
    references. The constructor's argument list mostly takes one of:
      - `unique_ptr.get()`  — a `CALL_EXPR` for `get`.
      - `&xxx_` for a non-owned pointer field (e.g. `&branch_tracker_`).
      - `xxx_.get()` directly.
    """
    field_name = _extract_field_name(arg)
    if field_name is not None:
        target = ctor_field_map.get(field_name)
        if target is not None:
            return target
    # Fall back to typing the expression.
    target = _unwrap_type_name(arg.type)
    if target in modules:
        return target
    return None


def _walk_pass_b(timing_ctor: cindex.Cursor,
                 timing_field_map: dict[str, str],
                 method_index: dict[str, dict[str, cindex.Cursor]],
                 modules: set[str]
                 ) -> tuple[list[Edge], list[str]]:
    """Pass B — wiring discovery.

    Walks `TimingModel::TimingModel`'s body for:
      - `wb_arbiter_->add_source(unit_.get())` — emits a REGISTERED edge
        from the unit to the arbiter (the unit produces results that the
        arbiter consumes).
      - `panic_->set_drained_query([this]() { return execution_units_drained(); })`
        — descends into the lambda, finds the helper call, then walks
        the helper's body for unit accessors. Each unit gets a
        REGISTERED back-pressure edge to the panic controller (the
        callable invokes only committed-state accessors).

    Other `set_*` setters wire member pointers but the directional
    semantics differ case-by-case; they are not turned into edges here.
    Pass C harvests those same wirings indirectly via
    `timing_field_map`.
    """
    edges: list[Edge] = []
    warnings: list[str] = []
    Kind = cindex.CursorKind

    for call in timing_ctor.walk_preorder():
        if call.kind != Kind.CALL_EXPR:
            continue
        spelling = call.spelling
        if spelling != "add_source" and not (spelling and spelling.startswith("set_")):
            continue
        children = list(call.get_children())
        if not children:
            continue
        callee = children[0]
        # Only handle calls of the form `member_->method(...)`.
        if callee.kind != Kind.MEMBER_REF_EXPR:
            continue
        callee_children = list(callee.get_children())
        receiver_field = (_extract_field_name(callee_children[0])
                          if callee_children else None)
        if receiver_field is None:
            continue
        consumer = timing_field_map.get(receiver_field)
        if consumer is None:
            continue

        if spelling == "add_source":
            # add_source(producer_unit). Argument resolves to the unit.
            for arg in children[1:]:
                producer = _resolve_arg_module(arg, timing_field_map, modules)
                if producer and consumer and producer != consumer:
                    edges.append(Edge(
                        src=producer, dst=consumer,
                        classification="REGISTERED", label="result",
                    ))
            continue

        # set_drained_query: lambda body calls a helper which reads
        # current_busy()/next_has_result()/next_fifo_empty()/has_pending_work()
        # on several modules. Produce a REGISTERED back-pressure edge
        # from each to the consumer (the panic controller). Direction
        # axis (back-pressure) is derived from cluster topology by the
        # renderer; the cycle axis is REGISTERED because the callable
        # invokes committed-state accessors only.
        if spelling == "set_drained_query":
            helper_targets = _walk_drained_lambda(
                call, method_index,
                timing_field_map=timing_field_map,
                modules=modules,
            )
            for target in helper_targets:
                if target != consumer:
                    edges.append(Edge(
                        src=target, dst=consumer,
                        classification="REGISTERED", label="drained",
                    ))
            continue
        # Other `set_*` setters: wiring only, not drawn as Pass-B edges.

    return edges, warnings


def _walk_drained_lambda(call: cindex.Cursor,
                         method_index: dict[str, dict[str, cindex.Cursor]],
                         timing_field_map: dict[str, str],
                         modules: set[str]) -> set[str]:
    """For `panic_->set_drained_query([this]() { return helper(); })`:
    descend into the lambda body, find the called helper method, then
    walk the helper for unit accessors and return the set of module
    names referenced.
    """
    Kind = cindex.CursorKind
    timing_methods = method_index.get("TimingModel", {})
    targets: set[str] = set()
    for cursor in call.walk_preorder():
        if cursor.kind != Kind.CALL_EXPR:
            continue
        callee_name = cursor.spelling
        if not callee_name:
            continue
        # Helper calls are member calls on `this` (no explicit
        # receiver). Only TimingModel's own private helper qualifies.
        helper_def = timing_methods.get(callee_name)
        if helper_def is None:
            continue
        for sub_call in helper_def.walk_preorder():
            if sub_call.kind != Kind.CALL_EXPR:
                continue
            sub_callee = list(sub_call.get_children())
            if not sub_callee:
                continue
            ref = sub_callee[0]
            if ref.kind != Kind.MEMBER_REF_EXPR:
                continue
            # Skip nested helper calls; we only want member calls of the
            # form `unit_->ready_out()`. Member-ref-expr base is the
            # field; resolve via timing_field_map.
            ref_children = list(ref.get_children())
            base_field = (_extract_field_name(ref_children[0])
                          if ref_children else None)
            if base_field is None:
                continue
            target = timing_field_map.get(base_field)
            if target and target in modules:
                targets.add(target)
    return targets


# ----------------------------------------------------------------------------
# Pass C — edge discovery from evaluate()/commit() bodies
# ----------------------------------------------------------------------------

def _is_const_method(cursor: cindex.Cursor) -> bool:
    """True if `cursor` references a const member function."""
    if cursor is None:
        return False
    referenced = cursor.referenced if cursor.kind == cindex.CursorKind.CALL_EXPR else cursor
    if referenced is None:
        return False
    try:
        return bool(referenced.is_const_method())
    except Exception:
        return False


# Method-name keywords that indicate a const accessor reading committed
# producer state. Under the two-axis model these all classify as
# REGISTERED on the cycle axis; the back-pressure direction overlay is
# derived from cluster topology by the renderer rather than from the
# accessor's name. Pre-rename names (`ready_out`, `is_*`) coexist with
# post-rename names (`current_*`, `next_*`) here so the extractor works
# across the Phase 1 transition.
KNOWN_ACCESSOR_NAMES = (
    "ready_out", "ready_to_consume_fetch", "is_idle", "has_result",
    "fifo_empty", "fifo_front", "is_stalled", "is_in_flight",
    "is_pending", "is_active", "is_busy", "has_pending_work",
    "has_response", "has_pending", "pending_warp", "pending_fill",
    "committed_entry", "current_output", "current_redirect_request",
    "current_ebreak_request",
)
KNOWN_ACCESSOR_RE = re.compile(
    r"^(" + "|".join(re.escape(n) for n in KNOWN_ACCESSOR_NAMES) + r")$"
)


def _extract_field_name(node: cindex.Cursor) -> str | None:
    """Return the field name reached by drilling through unexposed-expr
    and unary-operator wrappers around a member-ref base.

    Common AST shapes that appear in this codebase:
      - Pointer field `T* x_`: callee base is a `MEMBER_REF_EXPR` for
        `x_` directly.
      - Reference field `T& x_`: an `UNEXPOSED_EXPR` wraps the
        `MEMBER_REF_EXPR`. Drilling through the wrapper recovers it.
      - `unique_ptr<T> x_` accessed via `x_->method()`: the callee base
        is a `CALL_EXPR` for `operator->` whose first argument is the
        `MEMBER_REF_EXPR` for the field.
      - `unique_ptr<T> x_` accessed via `x_.get()`: a method-call chain
        whose receiver is the `MEMBER_REF_EXPR` for the field.

    The function returns the bare field spelling (e.g. `"alu_"`).
    `MEMBER_REF_EXPR` whose referenced cursor is a method (`get`,
    `operator->`) is followed into its receiver instead of being
    returned as the "field name".
    """
    Kind = cindex.CursorKind
    cur = node
    safety = 0
    while cur is not None and safety < 16:
        if cur.kind == Kind.MEMBER_REF_EXPR:
            # If the member ref is a method, drill into its receiver.
            referenced = cur.referenced
            ref_kind = referenced.kind if referenced is not None else None
            if ref_kind in (Kind.CXX_METHOD, Kind.FUNCTION_DECL):
                children = list(cur.get_children())
                if not children:
                    return None
                cur = children[0]
                safety += 1
                continue
            return cur.spelling
        if cur.kind in (Kind.UNEXPOSED_EXPR, Kind.PAREN_EXPR,
                        Kind.UNARY_OPERATOR, Kind.CSTYLE_CAST_EXPR,
                        Kind.CXX_FUNCTIONAL_CAST_EXPR,
                        Kind.CXX_STATIC_CAST_EXPR):
            children = list(cur.get_children())
            if not children:
                return cur.spelling or None
            cur = children[0]
            safety += 1
            continue
        if cur.kind == Kind.CALL_EXPR:
            # Method-call chain (e.g. `unique_ptr->`, `xxx_.get()`):
            # follow the call's first child to find the receiver.
            children = list(cur.get_children())
            if not children:
                return None
            cur = children[0]
            safety += 1
            continue
        if cur.kind == Kind.DECL_REF_EXPR:
            return cur.spelling
        return None
    return None


def _walk_method_body_for_edges(method_cursor: cindex.Cursor,
                                self_class: str,
                                self_field_map: dict[str, str],
                                self_methods: dict[str, cindex.Cursor],
                                modules: set[str],
                                depth: int = 0,
                                max_depth: int = 2,
                                visited: set[str] | None = None
                                ) -> list[tuple[str, str, str, str, bool]]:
    """Walk `method_cursor`'s body and return tuples of
    (src, dst, callee_name, label_hint, is_const_call).

    Same-class private helpers are inlined recursively up to
    `max_depth`. `visited` tracks helper names to break cycles.
    """
    if visited is None:
        visited = set()
    Kind = cindex.CursorKind
    out: list[tuple[str, str, str, str, bool]] = []

    for cursor in method_cursor.walk_preorder():
        # Same-class helper inlining.
        if cursor.kind == Kind.CALL_EXPR and depth < max_depth:
            callee_name = cursor.spelling
            if callee_name in self_methods and callee_name not in visited:
                helper = self_methods[callee_name]
                visited.add(callee_name)
                out.extend(_walk_method_body_for_edges(
                    helper, self_class, self_field_map, self_methods,
                    modules, depth + 1, max_depth, visited,
                ))
                continue

        # Member call: `member_->method()` or `member_.method()`.
        if cursor.kind == Kind.CALL_EXPR:
            children = list(cursor.get_children())
            if not children:
                continue
            ref = children[0]
            if ref.kind != Kind.MEMBER_REF_EXPR:
                continue
            ref_children = list(ref.get_children())
            field_name = (_extract_field_name(ref_children[0])
                          if ref_children else None)
            if field_name is None:
                continue
            target = self_field_map.get(field_name)
            if not target:
                continue
            if target == self_class:
                continue
            callee_name = ref.spelling
            referenced = ref.referenced
            is_const = False
            try:
                if referenced is not None:
                    is_const = bool(referenced.is_const_method())
            except Exception:
                pass
            label_hint = callee_name
            if is_const or KNOWN_ACCESSOR_RE.match(callee_name or ""):
                # READ: edge from producer to this consumer.
                out.append((target, self_class, callee_name, label_hint, True))
            else:
                # WRITE / mutate: edge from this producer to target consumer.
                out.append((self_class, target, callee_name, label_hint, False))
            continue

        # Field read: `member_.field`. The MEMBER_REF_EXPR sits on the
        # right of a MEMBER_REF_EXPR-based base. We only care when the
        # field name is `current_*` or `next_*` and the parent is not a
        # call (which we already handled above).
        if cursor.kind == Kind.MEMBER_REF_EXPR:
            name = cursor.spelling or ""
            if not (name.startswith("current_") or name.startswith("next_")):
                continue
            ref_children = list(cursor.get_children())
            base_field = (_extract_field_name(ref_children[0])
                          if ref_children else None)
            if base_field is None:
                continue
            target = self_field_map.get(base_field)
            if not target or target == self_class:
                continue
            # Field READ: edge from producer to this consumer.
            out.append((target, self_class, name, name, True))

    return out


def _walk_pass_c(modules: set[str],
                 method_index: dict[str, dict[str, cindex.Cursor]],
                 field_type_maps: dict[str, dict[str, str]]
                 ) -> tuple[list[Edge], list[str]]:
    """Pass C — edge discovery.

    For every module, walk its `evaluate()` and `commit()` bodies and
    collect cross-module member calls and `current_*`/`next_*` reads.
    Same-class private helpers come straight from `method_index` (which
    already covers in-class and out-of-line definitions); no per-class
    TU re-walk is needed.
    """
    edges: list[Edge] = []
    warnings: list[str] = []

    raw: list[tuple[str, str, str, str, bool]] = []
    for class_name in modules:
        helpers = method_index.get(class_name, {})
        field_map = field_type_maps.get(class_name, {})
        for entry_name in ("evaluate", "commit"):
            method = helpers.get(entry_name)
            if method is None:
                continue
            raw.extend(_walk_method_body_for_edges(
                method,
                self_class=class_name,
                self_field_map=field_map,
                self_methods=helpers,
                modules=modules,
            ))

    # Dedup per (src, dst, callee) — the same call shows up in both the
    # outer body and any helper inlining hits.
    seen: set[tuple[str, str, str]] = set()
    for (src, dst, callee, label, is_const) in raw:
        key = (src, dst, callee or "")
        if key in seen:
            continue
        seen.add(key)
        if src == dst:
            continue
        classification = _classify_edge(src, dst, callee, is_const)
        edges.append(Edge(
            src=src, dst=dst,
            classification=classification,
            label=_humanize_label(src, dst, callee),
        ))
    return edges, warnings


# ----------------------------------------------------------------------------
# Pass D — classification + labels
# ----------------------------------------------------------------------------

# Hand-curated classification overrides for cases where Pass D's
# heuristic disagrees with the documented discipline. Under the two-axis
# model the cycle axis takes only REGISTERED or COMBINATIONAL; the
# back-pressure direction overlay is a render-time concern.
#
# After Phase 10's REGISTERED conversions the only surviving carve-out
# is the gather-buffer write-port arbitration:
#
#   cache → gather_file `try_write` — COMBINATIONAL same-tick port
#   arbitration (row 11). `try_write` is a non-const mutating call, so
#   Pass D's name heuristic would tag it REGISTERED; in fact it mutates
#   `next_port_claimed_` and observes the live shared flag for
#   first-writer-wins arbitration within a single tick.
#
# The pre-Phase-10 entries are gone because the edges they covered are
# now genuinely REGISTERED and classify correctly without an override:
#   - LdSt → coalescing addr-gen FIFO is REGISTERED (Phase M1).
#   - mem_if → cache response poll reads `current_has_response()`
#     (Phase M5 rename); a `current_*` accessor classifies REGISTERED.
#   - cache → coalescing stall is now `next_cmd_ready()` (Phase M3); a
#     `next_*` callee classifies COMBINATIONAL from the name alone.
EDGE_CLASSIFICATION_OVERRIDES: dict[tuple[str, str, str], str] = {
    ("L1Cache",   "LoadGatherBufferFile", ""):       "COMBINATIONAL",
}

# Hand-curated label overrides for edges where the auto-derived label
# reads badly. Keyed by (src, dst, callee).
EDGE_LABEL_OVERRIDES: dict[tuple[str, str, str], str] = {
    # `note_redirect_applied` reads more naturally as the PR-style noun.
    ("FetchStage", "BranchShadowTracker", "note_redirect_applied"):
        "redirect applied",
    ("WarpScheduler", "BranchShadowTracker", "note_branch_issued"):
        "branch issued",
    ("Scoreboard", "WarpScheduler", "is_pending"): "scoreboard",
    ("Scoreboard", "WarpScheduler", "current_pending"): "scoreboard",
    ("WritebackArbiter", "Scoreboard", "clear_pending"): "scoreboard",
    ("WarpScheduler", "Scoreboard", "set_pending"): "scoreboard",
    # The Phase-3 method `current_redirect_request_or_override` produces
    # a verbose auto-label; trim to the architectural noun.
    ("OperandCollector", "FetchStage", "current_redirect_request_or_override"):
        "redirect",
    ("OperandCollector", "DecodeStage", "current_redirect_request_or_override"):
        "redirect",
}

# Manual edges that the AST can't infer because the calls live in
# orchestration methods (`TimingModel::tick`, `record_cycle_trace`,
# panic-flush cascade) rather than in any module's `evaluate()` /
# `commit()` body. After Phase 3 of the naming-and-access-discipline
# refactor inlined the unit-busy lambda and moved `read_redirect_request`
# onto `OperandCollector`, the only edges that still need manual
# attribution are these orchestrator-level ones.
#
#   - Panic-flush cascade: TimingModel makes the actual `flush()` calls
#     gated by `pending_panic_flush_`, but the conceptual source is
#     PanicController arming the flush via its evaluate() drain.
#   - DecodeStage/L1Cache -> TimingModel: TimingModel reads
#     `decode_->current_ebreak_request()` and cache trace events from
#     `tick()` and `record_cycle_trace()` respectively; we don't walk
#     TimingModel's orchestration methods because the bulk of their
#     calls are evaluate()/commit() lifecycle calls that would
#     over-generate edges.
#   - LdStUnit -> WritebackArbiter: registered as a writeback source
#     via `wb_arbiter_->add_source` in TimingModel's constructor; Pass
#     B picks up alu/mul/div/tlookup/gather_file via that mechanism but
#     not ldst (whose result also surfaces through addr_gen FIFO into
#     coalescing, complicating Pass B's heuristic).
MANUAL_AST_EDGES: list[tuple[str, str, str, str]] = [
    # (src, dst, classification, label)
    ("PanicController", "WarpScheduler",        "REGISTERED", "flush"),
    ("PanicController", "OperandCollector",     "REGISTERED", "flush"),
    ("PanicController", "LoadGatherBufferFile", "REGISTERED", "flush"),
    ("PanicController", "WritebackArbiter",     "REGISTERED", "flush"),

    ("DecodeStage", "TimingModel", "REGISTERED", "ebreak"),
    ("L1Cache",     "TimingModel", "REGISTERED", "trace events"),

    ("LdStUnit", "WritebackArbiter", "REGISTERED", "result"),
]
# Note: branch resolution moved into `ALUUnit::evaluate()` in Phase 10A
# (`note_resolved_correctly`), so the `ALUUnit → BranchShadowTracker`
# edge is now discovered by Pass C directly. The pre-10A manual
# `OperandCollector → BranchShadowTracker` edge has been removed —
# OperandCollector no longer writes the tracker.


def _classify_edge(src: str, dst: str, callee: str, is_const: bool) -> str:
    """Pass D classification.

    Under the two-axis model, the cycle axis takes only REGISTERED or
    COMBINATIONAL. Direction (forward-data / back-pressure) is an
    architectural overlay derived from cluster topology by the renderer.

    Priority order:
      1. EDGE_CLASSIFICATION_OVERRIDES exact-match (for documented
         COMBINATIONAL carve-outs that name-based inference can't
         distinguish from REGISTERED reads).
      2. Callee name starts with `next_` → COMBINATIONAL.
      3. Default → REGISTERED. Includes const-accessor reads (which
         used to be tagged READY/STALL) — back-pressure semantics are
         expressed by cluster topology, not by classification.
    """
    callee = callee or ""
    override = (EDGE_CLASSIFICATION_OVERRIDES.get((src, dst, callee))
                or EDGE_CLASSIFICATION_OVERRIDES.get((src, dst, "")))
    if override:
        return override
    if callee.startswith("next_"):
        return "COMBINATIONAL"
    return "REGISTERED"


def _humanize_label(src: str, dst: str, callee: str) -> str:
    """Pass D label generation.

    Strips `note_`/`set_`/`get_`/`current_`/`next_` prefixes, splits
    snake_case on underscores, and joins with a single space. Falls
    back to the raw callee name for unknown shapes. EDGE_LABEL_OVERRIDES
    wins over the auto-generated form.
    """
    override = EDGE_LABEL_OVERRIDES.get((src, dst, callee))
    if override:
        return override
    if not callee:
        return ""
    name = callee
    for prefix in ("note_", "set_", "get_", "current_", "next_", "is_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace("_", " ")


# ----------------------------------------------------------------------------
# Translation-unit parsing
# ----------------------------------------------------------------------------

def _parse_translation_units(entries: list[dict]
                             ) -> tuple[list[cindex.TranslationUnit], list[str]]:
    """Parse each translation unit and return the TUs plus diagnostics."""
    index = cindex.Index.create()
    tus: list[cindex.TranslationUnit] = []
    warnings: list[str] = []
    for entry in entries:
        argv = _split_command(entry)
        file_arg = entry["file"]
        directory = entry.get("directory", os.getcwd())
        if not os.path.isabs(file_arg):
            file_arg = str(Path(directory) / file_arg)
        try:
            tu = index.parse(
                file_arg,
                args=argv,
                options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
            )
        except cindex.TranslationUnitLoadError as exc:
            warnings.append(f"failed to parse {file_arg}: {exc}")
            continue
        for diag in tu.diagnostics:
            if diag.severity >= cindex.Diagnostic.Error:
                warnings.append(
                    f"{file_arg}: {diag.spelling} ({diag.location})"
                )
        tus.append(tu)
    return tus, warnings


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def extract(compile_commands: Path, sim_root: Path) -> ExtractionResult:
    """Run the AST extractor against the timing TUs in `compile_commands`."""
    entries = _load_compile_db(compile_commands)
    timing_entries = _select_timing_tus(entries, sim_root)
    if not timing_entries:
        return ExtractionResult(
            modules=[],
            edges=[],
            warnings=[
                f"no timing translation units found under "
                f"{sim_root}/src/timing in {compile_commands}"
            ],
        )

    tus, parse_warnings = _parse_translation_units(timing_entries)
    if not tus:
        return ExtractionResult(
            modules=[], edges=[],
            warnings=parse_warnings + [
                "no translation units parsed; check that the libclang "
                "version matches the compiler used to populate "
                "compile_commands.json"
            ],
        )

    seen_classes, base_map, decl_map, method_index, timing_ctor = _build_index(
        tus, sim_root,
    )

    modules, discovery_warnings = _discover_modules(seen_classes, base_map)
    module_names = {m.name for m in modules}

    field_type_maps = _build_field_type_map(decl_map, module_names)

    timing_field_map = field_type_maps.get("TimingModel", {})
    pass_b_edges: list[Edge] = []
    pass_b_warnings: list[str] = []
    if timing_ctor is not None:
        pass_b_edges, pass_b_warnings = _walk_pass_b(
            timing_ctor, timing_field_map, method_index, module_names,
        )
    else:
        pass_b_warnings = ["Pass B: TimingModel constructor definition not found"]

    pass_c_edges, pass_c_warnings = _walk_pass_c(
        module_names, method_index, field_type_maps,
    )

    # Manual augmentations for edges the AST can't infer (lambda
    # captures, free-function helpers, orchestration calls).
    manual_edges = [
        Edge(src=s, dst=d, classification=c, label=lbl)
        for (s, d, c, lbl) in MANUAL_AST_EDGES
        if s in module_names and d in module_names
    ]

    # Merge edges, dedup on (src, dst, classification).
    merged: dict[tuple[str, str, str], Edge] = {}
    for e in pass_b_edges + pass_c_edges + manual_edges:
        key = (e.src, e.dst, e.classification)
        if key not in merged:
            merged[key] = e
        else:
            existing = merged[key]
            if e.label and existing.label and e.label != existing.label:
                # De-dup label fragments split by " / " to keep merged
                # edges from accumulating duplicates across passes.
                fragments = [f.strip() for f in existing.label.split("/")]
                if e.label not in fragments:
                    existing.label = f"{existing.label} / {e.label}"

    warnings = (parse_warnings + discovery_warnings
                + pass_b_warnings + pass_c_warnings)
    warnings.insert(
        0,
        f"AST extractor: parsed {len(tus)} TUs -> "
        f"{len(modules)} modules, {len(merged)} edges",
    )

    return ExtractionResult(
        modules=modules,
        edges=list(merged.values()),
        warnings=warnings,
    )
