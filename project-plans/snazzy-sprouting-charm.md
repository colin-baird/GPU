# Plan: AST-derived signal-flow diagram

## Context

`tools/render_signal_diagram.py` currently parses
`resources/timing_discipline.md`'s "Per-boundary inventory" table to drive
the architecture poster. That works, but every new module or cross-stage
edge requires a hand-edit of the table. We want the diagram to update
automatically when a new `PipelineStage` / `ExecutionUnit` subclass
appears, when a new wiring setter is added, or when a new method call
shows up in an `evaluate()` body. The C++ source becomes the source of
truth; the markdown table stays as an authored cross-check.

The decision is to use **libclang AST** as the parser (full semantic
analysis, including the lambda-wired `panic_->set_drained_query(...)`
case) and **keep the markdown extractor available** for cross-checking
(`--source=markdown` and `--validate`).

`compile_commands.json` is already exported by CMake (verified at
`build/compile_commands.json`), so no build-system changes are needed.

## File layout

Refactor into four files under `tools/`:

| File | Responsibility |
|------|----------------|
| `tools/diagram_types.py` | `@dataclass Module(name, cluster)` and `@dataclass Edge(src, dst, classification, label, source_row=None)`. Imported by both extractors and the renderer. |
| `tools/diagram_extract_md.py` | The current markdown parser, lifted as-is and reshaped to expose `extract(md_path: Path) -> ExtractionResult`. |
| `tools/diagram_extract_ast.py` | New libclang extractor. Exposes `extract(compile_commands: Path, sim_root: Path) -> ExtractionResult`. |
| `tools/render_signal_diagram.py` | Slimmed down. Keeps presentation (`MODULES` cluster map, `CLUSTER_COLOR`, `NODE_LABEL`, `EDGE_LABEL_OVERRIDES`, `emit_dot`, `emit_mermaid`). New flags: `--source={ast,markdown}` (default `ast`), `--validate`, `--check`. |

A `tools/requirements.txt` file is added containing only `clang>=14`.
The script imports `clang.cindex` lazily and falls back to a clear error
if the import fails (with a one-line install hint), rather than crashing
at module-load time.

## Extraction algorithm (AST extractor)

Four passes over the translation units listed in `compile_commands.json`,
filtered to `sim/src/timing/*.cpp` and the corresponding headers under
`sim/include/gpu_sim/timing/`.

**Pass A — Module discovery.** Walk every `CXXRecordDecl`. A class is a
module if it transitively inherits `PipelineStage` or `ExecutionUnit`,
**or** appears in a small Python-side `STANDALONE_MODULES` set:
`{TimingModel, Scoreboard, BranchShadowTracker, CoalescingUnit, L1Cache,
ExternalMemoryInterface, PanicController}`. The standalone set is
presentation-driven and stable; not worth auto-detecting.

**Pass B — Wiring discovery.** Inside `TimingModel`'s constructor body
(see `sim/src/timing/timing_model.cpp:127–208`), collect every
`CXXMemberCallExpr` whose callee name matches `^set_` or is `add_source`.
Resolve the receiver's type (the consumer module) and each argument's
type (the producer module). For
`scheduler_->set_consumers(opcoll_, alu_, mul_, ...)` this yields five
producer→consumer wires from one statement. For
`panic_->set_drained_query([this]() { return execution_units_drained(); })`
descend into the `LambdaExpr`, find the `execution_units_drained`
private-method call, then recurse into that method's body to harvest
the upstream module accessors (cap recursion at depth 2).

**Pass C — Edge discovery.** For every module's `evaluate()` and
`commit()` method body, collect:
- `CallExpr`s whose receiver is a member pointer that resolves to
  another module → emit an edge. Direction depends on what the call
  does (see Pass D).
- `MemberExpr`s reading another module's `next_*` or `current_*` field
  directly → emit an edge with classification baked in.

When the `evaluate()` body calls a private helper of the same class,
inline that helper's body (depth 2) so e.g. `CoalescingUnit`'s helper
calls to `cache_.process_load(...)` are attributed to the right edge.

**Pass D — Classification.** Per edge, in priority order:
1. Callee name matches `^ready_out$|^ready_to_consume|^is_idle$|^has_result$|^fifo_empty$|^is_stalled$|^is_in_flight$|^is_pending$|^is_active$` → **READY/STALL**.
2. Else AST resolves the callee to a `const` method → **READY/STALL** (catches future const accessors without hardcoding names).
3. Else the consumer reads a producer's `current_*` field → **REGISTERED**.
4. Else the consumer reads a producer's `next_*` field, OR the call mutates the producer (non-const, non-`set_*` method) → **COMBINATIONAL**.
5. Else → unclassified; emit warning, render as a thin grey edge so the gap is visible.

Rule 4 will misclassify a few REGISTERED-via-callee cases (e.g.,
`WritebackArbiter::evaluate` calls `scoreboard_.clear_pending()` which
internally writes only `next_*`). Resolve via
`EDGE_CLASSIFICATION_OVERRIDES: dict[(producer, consumer, method), str]`
in the renderer for the handful of documented exceptions.

**Label generation.** Method/field name → human label by:
- Strip `note_`/`set_`/`get_`/`current_`/`next_` prefix.
- Split snake_case on `_`, join with space.
- E.g., `note_branch_issued` → "branch issued"; `current_redirect_request` → "redirect request"; `is_stalled` → "stalled".
- Fall back to receiver's role for generic verbs (`accept` → "dispatch", `evaluate`/`commit` not labels).
- An `EDGE_LABEL_OVERRIDES: dict[(src, dst, method), str]` covers the
  ~5–10 cases where the auto label reads badly.

## Manual configuration that remains

These are presentation choices that the AST has no opinion about and
stay as Python literals in `render_signal_diagram.py`:

- `MODULES` (cluster assignment per module) — Frontend & Issue / Execute / Memory / Writeback / Control.
- `CLUSTER_COLOR`.
- `NODE_LABEL` (multi-line PascalCase splits, aesthetic line breaks).
- `STANDALONE_MODULES` set in the extractor (the 7 control classes outside the two hierarchies).
- `EDGE_LABEL_OVERRIDES` (small).
- `EDGE_CLASSIFICATION_OVERRIDES` (small; documented carve-outs like cache MSHR/tag mutations and DRAMSim3 queue writes that intentionally don't follow `next_*`/`current_*`).

## Phased migration

1. **Phase 0 — Refactor only, no AST yet.** Extract `diagram_types.py` and `diagram_extract_md.py` from `render_signal_diagram.py`. Renderer goes through `Module`/`Edge` dataclasses. Output is byte-for-byte identical to today's `tools/signal_diagram.dot`. Risk-free; first runnable milestone.
2. **Phase 1 — AST extractor skeleton.** `diagram_extract_ast.py` implements Pass A only (module discovery). Add `tools/requirements.txt`. CLI gains `--source=ast`. Verify the discovered module set matches the markdown extractor (should produce 5 + 7 + 7 = 19 modules).
3. **Phase 2 — Wiring + edges.** Implement Passes B and C. Edges emitted without classification (single style). Add `--validate` mode that runs both extractors and prints set differences keyed by `(src, dst)`.
4. **Phase 3 — Classification + labels.** Implement Pass D and the label generator. Iterate `EDGE_LABEL_OVERRIDES` and `EDGE_CLASSIFICATION_OVERRIDES` until the rendered SVG looks right (not necessarily byte-identical to today's — auto labels will differ but should be sane).
5. **Phase 4 — Flip default.** `--source=ast` becomes the default. Markdown extractor stays for `--source=markdown` and `--validate`.

## Validation

- **`--validate` mode.** Runs both extractors, reports: (a) modules present in one but not the other; (b) edges differing by `(src, dst, classification)` triple, ignoring labels; (c) classification disagreements. Phase 4 promotes this to a CI gate that fails on differences except those in a curated allow-list (e.g., AST naturally finds the cache↔mem_if internal edges that markdown row 15 treats as a documented carve-out).
- **Snapshot test** at `tests/test_signal_diagram.py`: imports the extractor, runs against `build/compile_commands.json`, asserts module count is exactly 19 and a hand-curated set of ~10 known edges appear with the expected classification (e.g., "FetchStage reads `DecodeStage::ready_to_consume_fetch()`" → READY/STALL; "WarpScheduler writes `BranchShadowTracker.note_branch_issued`" → REGISTERED).
- **Visual diff.** Each phase PR includes before/after `signal_diagram.svg` rendered side-by-side. The poster itself is the highest-bandwidth correctness signal.

## Critical files to read / modify

To **read**:
- `sim/include/gpu_sim/timing/pipeline_stage.h` — base hierarchy.
- `sim/include/gpu_sim/timing/execution_unit.h` — second base hierarchy and the canonical `ready_out()` / `has_result()` / `consume_result()` shapes.
- `sim/src/timing/timing_model.cpp:127–208` — wiring section the AST walks for Pass B.
- `sim/src/timing/warp_scheduler.cpp` (`evaluate`), `sim/src/timing/coalescing_unit.cpp` (`evaluate`), `sim/src/timing/writeback_arbiter.cpp` (`evaluate`) — representative edge-discovery cases for Pass C.
- `resources/timing_discipline.md` — discipline doc; the inventory table remains authoritative for cross-check.
- `build/compile_commands.json` — already generated by CMake; first-run check.
- `tools/bench_compare.py` — Python tooling style precedent.

To **modify / create**:
- `tools/render_signal_diagram.py` (refactor; slim down).
- `tools/diagram_types.py` (new, ~30 lines).
- `tools/diagram_extract_md.py` (new, lifted from current renderer).
- `tools/diagram_extract_ast.py` (new, the bulk of the work).
- `tools/requirements.txt` (new; one line: `clang>=14`).
- `tests/test_signal_diagram.py` (new snapshot test).
- `.claude/CLAUDE.md` Project Structure block — note the new files under `/tools/`.
- `resources/timing_discipline.md` footer — extend the existing pointer to mention `--source=ast` is now primary and `--validate` cross-checks against the table.

## Trickiest parts (acknowledged)

- **Lambda wiring** (`panic_->set_drained_query([this]() { return execution_units_drained(); })`) needs a two-hop AST walk: lambda body → private method call → that method's body → upstream accessors. Without it, all Control-cluster edges from the execution units are missed.
- **`evaluate()` private helpers** (e.g., `CoalescingUnit` factors logic into helpers that touch `cache_`/`ldst_`/`gather_file_`). Pass C must inline same-class helper bodies to depth ~2 or it under-counts edges. Beyond depth 2, emit a warning.
- **Member-pointer type resolution** depends on `compile_commands.json` carrying the right `-I` flags. Detect a missing or stale compile DB up front and print an actionable error, not a confusing "couldn't resolve `OperandCollector`" stream.
- **Same-tick vs. cross-tick mutation classification** is the heuristic most likely to need overrides (`scoreboard_.clear_pending()` is REGISTERED via callee internals, but Pass D rule 4 will tag it COMBINATIONAL by default). Accept the override list; document the specific rows that hit this.
- **Cache↔mem_if and cache internal carve-outs** (`timing_discipline.md` rows 10, 15) are intentional discipline exceptions. Either suppress these edges via `EDGE_CLASSIFICATION_OVERRIDES` or render with a distinct "carve-out" style. Pick the latter — keeps the diagram honest about where the discipline doesn't apply.

## Verification (end-to-end)

After each phase, run:
```
cmake -B build && cmake --build build -j8     # ensure compile_commands.json is fresh
python3 tools/render_signal_diagram.py --source=ast --svg
python3 tools/render_signal_diagram.py --validate
pytest tests/test_signal_diagram.py
```

The `--svg` render must produce a non-empty `tools/signal_diagram.svg`,
the `--validate` run must report no differences in module sets and only
allow-listed edge differences, and the snapshot test must pass.
