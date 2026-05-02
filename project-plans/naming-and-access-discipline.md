# Plan: Naming and access-pattern discipline for the timing model

## Context

The libclang-based signal-flow extractor (`tools/diagram_extract_ast.py`)
landed with ~150 lines of override and manual-edge tables. Most of those
exist because the C++ timing model uses inconsistent naming for the same
class of signal (six different shapes for READY/STALL accessors), three
different field-access encodings for the same logical "this module
holds a reference to that one" relationship, and a handful of helper /
lambda patterns that hide cross-stage reads from static analysis.

The goal is to drive the override and manual-edge tables to near-empty
by making the C++ source self-describing: a method's name and access
pattern should be enough to determine its discipline classification.

This is a behavior-preserving refactor. No cycle counts change. No new
features. Workload benchmarks must remain byte-identical to the current
baseline at every phase boundary.

## Decisions

### Replace the three-way classification with a two-axis model

The current `timing_discipline.md` defines three classifications
(REGISTERED, COMBINATIONAL, READY/STALL). This collapses two
independent axes into one inconsistently:

- **Cycle axis** — when is the read sampled? *Committed* state
  (producer's `current_*`, stable through evaluate phase) vs. *live*
  state (producer's `next_*`, mid-tick, ordering-sensitive).
- **Direction axis** — what does the edge mean in the architecture
  diagram? *Forward-data* (consumer reads producer; payload moves
  upstream→downstream) vs. *back-pressure* (producer reads consumer's
  "are you ready?" signal; the edge in the diagram points
  consumer→producer, against data flow).

The four cells map to mechanically distinct read patterns only on the
cycle axis. `OperandCollector::ready_out()` (back-pressure +
committed) and `Scoreboard::is_pending()` (forward-data + committed)
are mechanically identical — both are const accessors returning
committed state — but the existing taxonomy classifies them
differently (READY/STALL vs. REGISTERED), forcing the AST extractor's
override list to disambiguate. Symmetrically, `L1Cache::is_stalled()`
(back-pressure + live) is classified COMBINATIONAL, lumped with
forward-data live reads.

The fix is to make C++ naming track only the cycle axis. The
direction axis stays as architectural metadata and drives
visualization styling in the diagram, not signal type.

### Two prefixes, one per cycle discipline

| Prefix | Cycle discipline | Read semantics |
|--------|------------------|----------------|
| `current_*()` | **REGISTERED** | const accessor returning the producer's `current_*` slot (committed state, flipped at end-of-cycle by `commit()`). Stable through the entire evaluate phase regardless of where in the sweep it is queried. |
| `next_*()` | **COMBINATIONAL** | const accessor returning the producer's `next_*` slot (live mid-tick value). Ordering-sensitive; call-site comment required to identify which producer must run first. |

That's all. No `ready_*()` prefix — the back-pressure idiom is
expressed through the same two prefixes (a back-pressure signal is
just a `current_*()` accessor on the consumer that the producer reads
during its own evaluate). The diagram's dotted-blue back-pressure edge
style is derived from architectural cluster topology, not from
accessor names.

Methods that don't fit (constructors, lifecycle hooks, payload
mutations, internal helpers) keep their existing names. The prefix
rule applies only to **cross-stage const reads**.

### Postfix design language

After the cycle prefix, every accessor falls into one of three
grammatical shapes. The shape is derived from what the accessor
returns, and the chosen vocabulary should be specific enough to read
as English at the call site.

| Shape | Returns | Postfix grammar | Examples |
|-------|---------|-----------------|----------|
| **State predicate** | `bool` | `<prefix>_<adjective>` | `current_busy()`, `current_idle()`, `next_stalled()`, `current_in_flight(w)`, `current_pending(w, r)`, `next_fifo_empty()` |
| **Possession predicate** | `bool` | `<prefix>_has_<noun>` | `next_has_result()`, `next_has_response()` |
| **Payload accessor** | non-`bool` (`std::optional<T>`, payload struct, scalar id) | `<prefix>_<noun>` | `current_output()`, `current_pending_warp()`, `current_redirect_request()`, `current_ebreak_request()`, `next_fifo_front()` |

Rules:

1. **State predicates** describe a *condition* the producer is in.
   Use a bare adjective phrase, no `is_*` / `has_*` filler. Multi-word
   adjectives (`in_flight`, `fifo_empty`) split on underscores.
2. **Possession predicates** describe whether the producer *holds* a
   thing. Use the `has_<noun>` form because the question is about
   ownership, not state. Reserve this shape for accessors that are a
   precondition for a follow-up read of the actual thing
   (`if (next_has_result()) entry = consume_result();`).
3. **Payload accessors** return the thing itself. Bare noun. The
   prefix already conveys cycle discipline; the noun describes what's
   inside.
4. **Scope is carried by parameters, not name suffixes.**
   `current_busy(WarpId w)` instead of `current_busy_for_warp(w)`;
   `current_pending(WarpId w, RegIndex r)` instead of
   `current_pending_for_warp_register(w, r)`. The parameter list
   already names the scope axes.

The result reads naturally at the call site (`if
(opcoll_->current_busy()) skip;`, `if
(branch_tracker_->current_in_flight(w)) skip;`, `if
(unit_->next_has_result()) consume();`) and the prefix mechanically
encodes the cycle discipline.

### Single polarity convention: asserted = blocking

Every state predicate returns `true` when the *condition that
prevents forward progress* is in effect. The reader writes `if
(predicate) skip;` to bail out; no negation in the common case.

| Predicate | True means |
|-----------|------------|
| `current_busy()` | producer cannot accept more work this cycle |
| `current_in_flight(w)` | warp `w`'s branch shadow blocks issue |
| `current_pending(w, r)` | warp `w`'s register `r` is reserved (scoreboard hit) |
| `next_stalled()` | producer is stalling its consumer this cycle |
| `next_fifo_empty()` | producer's FIFO is empty (consumer cannot pop) |

Possession predicates have a distinct polarity rule because they
describe ownership, not blocking: `next_has_result()` returns `true`
when the result is available. This reads naturally as
"if-then-consume" rather than "if-blocked-then-skip". The two
polarities don't conflict because the shapes are different (state
adjective vs. `has_<noun>`).

The polarity rule eliminates inverse pairs. The codebase currently
has both `ready_out()` (true = ready) and `is_busy()` (true = busy)
for what is mechanically the same kind of signal. Under the
convention, only one form survives — `current_busy()` — and the
panic-drained query negates at the call site:

```cpp
const bool drained = !opcoll_->current_busy() &&
                     !alu_->current_busy() && !alu_->next_has_result() &&
                     ...
```

This is two characters more verbose than the existing
`opcoll_->ready_out() && ...` form, but pays it back at every
issue-gating site in the scheduler (where the negation moves from
five `!` operators to zero).

The convention is codified in `/resources/cpp_coding_standard.md` as
the canonical home, the corresponding section of
`/resources/timing_discipline.md` is rewritten around the two-axis
model, `.claude/CLAUDE.md` General Instructions points at the new
section, and `tools/lint_timing_naming.py` enforces the prefix /
postfix / polarity rules — all landed up-front in **Phase 0** before
any C++ change.

### One field-access shape per relationship type

`WarpScheduler` currently mixes references and pointers for dependencies
that have the same lifetime. Standardize per relationship:

| Lifetime | Holder type |
|----------|-------------|
| Owned | `std::unique_ptr<T>` (only `TimingModel` qualifies) |
| Mandatory at construction, never null | `T&` (constructor parameter) |
| Wired post-construction, may be null in tests | `T*` (`set_*` setter, `nullptr` default) |

The mix in `WarpScheduler` (refs for `scoreboard_`/`branch_tracker_`,
pointers for `opcoll_`/`alu_`/...) is migrated to all-pointer with a
single `set_dependencies()` setter. Test fixtures that rely on partial
wiring already use the override slots (`opcoll_ready_override_`, etc.);
no new test plumbing required.

### Cross-stage reads must have a static call site

Helpers, lambdas, and free functions that hide a cross-stage read inside
a parameter-bound call (so libclang can't resolve which module is on the
other end) are inlined or moved to methods on the appropriate module.
The two known offenders are `WarpScheduler::query_unit_ready` (lambda
parameterized over `ExecutionUnit*`) and the free function
`read_redirect_request(override, opcoll_)` called from
`FetchStage::commit` and `DecodeStage::commit`.

## File layout

One new file: `tools/lint_timing_naming.py`, the convention checker
introduced in Phase 0. Documentation lands in existing files
(`/resources/cpp_coding_standard.md`, `/resources/timing_discipline.md`,
`/resources/onboarding.md`, `.claude/CLAUDE.md`). All C++ changes land
in existing headers and `.cpp`s under `sim/include/gpu_sim/timing/` and
`sim/src/timing/`.

## Phasing

Each phase is independently regression-preserving. After every phase,
the full test suite and all six workload benchmarks must pass with
byte-identical cycle counts.

### Phase 0 — Codify the conventions in writing (must land first)

Documentation-only phase. No C++ change. Establishes the rules every
subsequent phase points back to and writes the lint that Phase 6 turns
into a CI gate. Must land before any rename so commits and code review
in Phases 1–5 have a stable reference to cite.

Six artifacts:

1. **`/resources/timing_discipline.md`** — *substantive rewrite*. This
   is the largest deliverable in Phase 0; the rest of Phase 0 points
   back to its sections.

   Replace the existing "Signal classifications" section (REGISTERED /
   COMBINATIONAL / READY/STALL three-way taxonomy) with the two-axis
   model:
   - "Cycle discipline" subsection covering REGISTERED (`current_*`)
     and COMBINATIONAL (`next_*`). Examples and the existing
     reference-implementation (`Scoreboard`) carry over.
   - "Edge direction" subsection introducing forward-data vs.
     back-pressure as orthogonal architectural metadata, **not** a
     signal classification. Worked example: `OperandCollector`
     exposes `current_busy()` (REGISTERED, `current_*` slot); the
     scheduler reads it during `scheduler.evaluate()`; the diagram
     renders the edge consumer→producer because the scheduler is
     downstream of opcoll in the data flow.

   Rewrite the per-boundary inventory table:
   - Drop the READY/STALL value from the Classification column. Every
     row's classification becomes one of REGISTERED, COMBINATIONAL, or
     a documented mixed pair.
   - Add a Direction column (or fold into the existing "Tick-order
     constraint" column) noting *forward-data* or *back-pressure* per
     row. Rows previously classified READY/STALL become REGISTERED +
     back-pressure.
   - Update accessor names in row prose to match the Phase 1 rename
     map.

   Rewrite the "Forbidden patterns" section to reflect the new model:
   - REGISTERED accessor not prefixed `current_*`.
   - COMBINATIONAL accessor not prefixed `next_*`.
   - Predicate that doesn't follow the postfix design language (no
     `is_*` filler on state predicates, no inverse-polarity twins
     where one already exists, etc.).
   - Cross-stage read inside a lambda body or free-function helper
     where the receiver is a parameter (statically opaque).
   - REGISTERED state field exposed as `public` (must go through an
     accessor).

   Add a "New module checklist" subsection enumerating what a
   contributor adding a new timing module must satisfy: cluster
   assignment in the diagram extractor, accessor prefix per the
   cycle axis, postfix shape per the design language, asserted=
   blocking polarity for state predicates, private REGISTERED state,
   an entry in the per-boundary inventory.

2. **`/resources/cpp_coding_standard.md`** — new section
   "Cross-stage accessor naming" that's the canonical compact
   reference for the rules:
   - Two prefixes (`current_*` / `next_*`) and which cycle discipline
     each maps to.
   - Postfix design language (state predicate / possession predicate /
     payload accessor) with one example each.
   - Polarity rule (asserted = blocking for state predicates;
     `has_<noun>` forms are positive).
   - Field-access shape rule (owned → `unique_ptr`,
     constructor-injected → reference, post-wired → pointer).
   - "No parameter-bound cross-stage reads" rule (no lambdas
     parameterized over module pointers; no free functions taking a
     module pointer to indirect a read; cross-stage reads must have a
     statically resolvable receiver).
   - Required private-field discipline for REGISTERED state.

   The coding-standard section is *concise*; the rationale and
   per-boundary inventory live in `timing_discipline.md`. Cross-link
   in both directions.

3. **`.claude/CLAUDE.md`** General Instructions adds a one-line
   pointer: "Cross-stage accessors in the timing model follow the
   prefix / postfix / polarity rules documented in
   `/resources/cpp_coding_standard.md` § Cross-stage accessor naming;
   the rationale and per-boundary inventory are in
   `/resources/timing_discipline.md`."

4. **`/resources/onboarding.md`** Code Conventions section gains a
   pointer to the same section. New contributors learn the rule on
   day one.

5. **`tools/lint_timing_naming.py`** — Python script that scans every
   header under `sim/include/gpu_sim/timing/` and reports violations.
   The lint enforces three layers:

   *Prefix layer* — every public const member function returning
   `bool`, `std::optional<…>`, or a payload reference must have a
   name matching `^(current|next)_`. Constructors, destructors, and
   lifecycle hooks (`evaluate`, `commit`, `reset`, `flush`,
   `seed_next`, `accept`, `consume_result`, `add_source`,
   `set_drained_query`, etc.) are exempt via an explicit allowlist.

   *Postfix layer* — for each accessor, classify by return type:
   - `bool` and name doesn't match `^(current|next)_has_` → expect
     state predicate; flag any `_is_` / `_has_` mid-name patterns
     that suggest the wrong shape.
   - `bool` and name matches `^(current|next)_has_` → possession
     predicate; OK.
   - non-`bool` → expect payload accessor; flag if the name contains
     adjective tells (`_busy`, `_idle`, `_stalled`, etc.).

   *Polarity layer* — for state predicates that participate in flow
   control (heuristic: called from a sibling module's `evaluate()`
   inside an `if` condition that early-returns), expect the asserted=
   blocking polarity. The lint flags pairs of accessors on the same
   class whose names are inverses (`busy` and `ready`, `stalled` and
   `advancing`, etc.) — only one polarity should exist per concept.
   This layer can produce false positives on edge cases; those get a
   per-line `// timing-naming-allow: <reason>` annotation.

   *Field-shape layer* — REGISTERED state fields (`current_*` /
   `next_*` named) must be `private`. Bare references to other module
   classes held as members are flagged as candidates for the post-
   wired-pointer rule (human decides).

   The script lives next to the diagram extractors so it shares the
   compile-database access pattern; libclang is already a project
   dependency.

   **Initial mode: report-only.** Phase 0 lands the script with `exit
   0` regardless of findings, since Phases 1–5 haven't been done yet
   and the existing source has many violations. The script prints a
   sorted list of violations to stdout for tracking. Phase 6 flips it
   to enforcement.

   The CMake test integration is wired in Phase 0 too (`add_test(NAME
   timing_naming_lint COMMAND python3 tools/lint_timing_naming.py
   --report-only)`) so the lint runs in CI from day one. The mode
   flip in Phase 6 is a one-line change.

6. **`tools/diagram_extract_md.py`** and
   **`tools/diagram_extract_ast.py`** — adjustment to the
   classification axis. Both extractors stop emitting "READY/STALL"
   as a classification value; instead the renderer derives the
   back-pressure render style from cluster topology
   (consumer→producer cluster ordering = back-pressure). The
   extractors emit only REGISTERED or COMBINATIONAL.
   `tools/render_signal_diagram.py` `EDGE_STYLE` table grows a
   computed-direction step that picks between the solid forward style
   and the dotted back-pressure style at render time.

**Phase 0 acceptance:**
- All six artifacts land in a single commit.
- Existing tests still pass (the lint is report-only, so no findings
  fail the build).
- The diagram renderer still produces valid SVG/DOT/Mermaid; the
  legend is updated to describe two cycle disciplines plus a
  direction overlay rather than three classifications.
- `python3 tools/render_signal_diagram.py --validate` continues to
  report zero unexpected differences — both extractors agree on the
  reduced classification axis.
- Running `python3 tools/lint_timing_naming.py` on the current source
  prints a non-empty list of violations — that list is the work
  Phases 1–5 work through.

### Phase 1 — Cross-stage accessor unification (mechanical rename + polarity flip)

**Highest leverage. Do this first.** Pure rename pass driven by the
Phase 0 conventions; no logic change other than the polarity flip on
back-pressure predicates.

Rename map, grouped by postfix shape:

**State predicates (return `bool`, asserted = blocking).** All
back-pressure and blocking predicates collapse to the `current_busy`
or `current_<adj>` form; existing inverse-polarity twins are dropped.

| Current | New | Cycle | Notes |
|---|---|---|---|
| `OperandCollector::ready_out()` | `current_busy()` | REGISTERED | polarity flip; consumer code becomes `if (current_busy()) skip;` |
| `DecodeStage::ready_to_consume_fetch()` | `current_busy()` | REGISTERED | polarity flip; same shape as opcoll |
| `LoadGatherBufferFile::is_busy(w)` | `current_busy(w)` | REGISTERED | no polarity change; drop `is_` filler |
| each `ExecutionUnit::ready_out()` (and per-unit overrides) | `current_busy()` | REGISTERED | polarity flip; uniform across the 5 units |
| `WritebackArbiter::has_pending_work()` | `current_busy()` | REGISTERED | polarity reframe (possession→state); the underlying meaning is "is the arbiter holding pending work?" which is a state predicate |
| `BranchShadowTracker::is_in_flight(w)` | `current_in_flight(w)` | REGISTERED | drop `is_` filler |
| `Scoreboard::is_pending(w, r)` | `current_pending(w, r)` | REGISTERED | drop `is_` filler |
| `L1Cache::is_stalled()` | `next_stalled()` | COMBINATIONAL | drop `is_` filler; cycle prefix flip captures it's a live read |
| `LdStUnit::fifo_empty()` | `next_fifo_empty()` | COMBINATIONAL | aggregate-state form `<noun>_<adj>` |

**Possession predicates (return `bool`, asserted = positive).**
Reserved for "does the producer hold this thing?" queries that
precede a payload consume.

| Current | New | Cycle | Notes |
|---|---|---|---|
| each `ExecutionUnit::has_result()` | `next_has_result()` | COMBINATIONAL | unchanged shape; cycle prefix added |
| `ExternalMemoryInterface::has_response()` | `next_has_response()` | COMBINATIONAL | unchanged shape; cycle prefix added |

**Payload accessors (return `std::optional<T>` or payload struct).**
Bare nouns; cycle prefix mandatory.

| Current | New | Cycle | Notes |
|---|---|---|---|
| `FetchStage::current_output()` | `current_output()` | REGISTERED | already conformant; no change |
| `DecodeStage::pending_warp()` | `current_pending_warp()` | REGISTERED | add cycle prefix |
| `DecodeStage::current_ebreak_request()` | `current_ebreak_request()` | REGISTERED | already conformant |
| `OperandCollector::current_redirect_request()` | `current_redirect_request()` | REGISTERED | already conformant |
| `LdStUnit::fifo_front()` | `next_fifo_front()` | COMBINATIONAL | add cycle prefix |
| `WritebackArbiter::committed_entry()` | `current_committed_entry()` | REGISTERED | add cycle prefix |

Polarity flips (the rows marked "polarity flip" above) require every
caller updated in the same commit. Sequence per accessor:
1. Rename the accessor *and* flip polarity at the same time, in one
   commit. Catch `[[maybe_unused]]` / unused-result warnings to find
   missed call sites. The compiler catches type errors but not
   inverted boolean uses, so a careful grep for the old name across
   the tree is mandatory before the commit lands.
2. Cycle counts must stay byte-identical. Run the workload benchmark
   suite as the gate; any non-zero diff indicates a missed inversion.

After this phase:

- `tools/diagram_extract_ast.py` `EDGE_CLASSIFICATION_OVERRIDES`
  collapses to empty. Pass D's classification reduces to
  `name.startswith('next_') -> COMBINATIONAL else REGISTERED`.
- `_humanize_label` produces cleaner labels: `current_busy` →
  "busy"; `current_in_flight` → "in flight"; `next_has_result` → "has
  result".
- `tools/diagram_extract_md.py` `ROW_OVERRIDES` for rows 5, 8, 9, 14
  can be simplified — both extractors emit only REGISTERED /
  COMBINATIONAL, so the row prose now produces correct
  classifications via the cross-product parser.
- `tools/render_signal_diagram.py` `EDGE_STYLE` carries the back-
  pressure styling decision; back-pressure detection uses
  `MODULE_CLUSTER` to identify edges where `dst` is upstream of
  `src` in the dataflow ordering.
- `/resources/timing_discipline.md` per-boundary inventory rows 1, 2,
  3, 4, 5, 8, 9, 11, 14, 15 are updated to use the new accessor
  names. Row 5 also gets a worked example of the postfix design
  language.

### Phase 2 — Field-access shape standardization

`WarpScheduler` is the only class that needs surgery. Convert
`scoreboard_` and `branch_tracker_` from references to pointers; add
them to the `set_dependencies()` setter (renamed from
`set_consumers()`). The constructor takes only `(num_warps, warps,
func_model, stats)`; everything else is wired after construction by
`TimingModel`.

Test fixtures that previously constructed a `WarpScheduler` with refs
need `set_dependencies()` calls; the override slots already exist for
the cases where the fixture doesn't wire real consumers.

After this phase, `_extract_field_name` in
`tools/diagram_extract_ast.py` no longer has to handle the
`UNEXPOSED_EXPR(MEMBER_REF_EXPR)` reference pattern — every receiver is
either a pointer (raw or smart) or `*this`. The helper drops to ~20
lines.

### Phase 3 — Lambda and free-function cleanup

Two surgical edits:

1. **Inline `WarpScheduler::query_unit_ready`'s `resolve` lambda** per
   case. The lambda parameterizes over `(override, ptr)`; the bodies
   become five copies of the override-fallback-true triple. Five
   ~3-line copies replace one lambda + switch — total line count goes
   up by ~10, but every cross-stage read has a static call site naming
   both endpoints.

2. **Move `read_redirect_request` onto `OperandCollector`.** Currently
   the free function in `redirect_request.h` takes
   `(override, OperandCollector*)` and returns the resolved request.
   Replace with `OperandCollector::current_redirect_request_or_override(
   const std::optional<RedirectRequest>& override) const`. `FetchStage`
   and `DecodeStage` call it directly; their override slots stay in
   place.

After this phase, `tools/diagram_extract_ast.py` `MANUAL_AST_EDGES`
loses every entry except the orchestrator-level edges (DecodeStage →
TimingModel, L1Cache → TimingModel, PanicController → flush targets)
that come from `TimingModel::tick()` itself rather than any
`evaluate()` / `commit()` body. Those would require walking
`TimingModel::tick()` with semantic awareness of which calls are
orchestration (skip) vs. observation (emit edge), which is out of scope
for this plan.

### Phase 4 — REGISTERED-writer naming consolidation

Lower-impact polish. Decide whether `note_*` (event-shape) or
`set_*`/`clear_*` (imperative) is the canonical naming for REGISTERED
writers. `BranchShadowTracker` is `note_*`; `Scoreboard` is
`set_*`/`clear_*`. Pick one and rename the other.

If `note_*`:
- `Scoreboard::set_pending` → `Scoreboard::note_dest_claimed`
- `Scoreboard::clear_pending` → `Scoreboard::note_dest_released`

If `set_*`/`clear_*`:
- `BranchShadowTracker::note_branch_issued` → `set_branch_in_flight`
- `BranchShadowTracker::note_resolved_correctly` → `clear_branch_in_flight`
- `BranchShadowTracker::note_redirect_applied` → `clear_branch_in_flight_post_redirect` (the third clear is event-distinct, so consolidating loses information)

The `note_*` direction is preferred — events are first-class, the
choice of `next_*` vs full reset is an implementation detail. But this
phase is optional and worth doing only when adjacent code is being
touched anyway.

### Phase 5 — Private-field enforcement at the public boundary

Audit pass. For every module, ensure REGISTERED state (`current_*` /
`next_*` fields) is `private` and exposed only via `current_*()` /
`next_*()` accessors. Some modules already follow this (`Scoreboard`);
others expose the field directly (`FetchStage::current_output_` is
`private` but `next_output_` is read by friends-style call patterns).

This phase is a guard rail for the future: a refactor that changes a
field to a method (or vice versa) silently changes the AST shape my
walker sees. Forcing accessors keeps the call site stable across
refactors.

After this phase, `tools/diagram_extract_ast.py`'s
`_walk_method_body_for_edges` loses the `MEMBER_REF_EXPR`-based field-
read branch entirely — every cross-stage read is a `CALL_EXPR`.

### Phase 6 — Flip the lint to enforcement

After Phases 1–5 land, `tools/lint_timing_naming.py` reports zero
violations on the current source. This phase makes future violations
fail the build.

Three changes:

1. **Flip the script default to enforcement.**
   `python3 tools/lint_timing_naming.py` (no flag) now exits 1 on any
   violation. The `--report-only` flag is retained for local
   exploration; CI invokes the script without it.

2. **Update the CMake test from `--report-only` to enforcement.**
   `add_test(NAME timing_naming_lint COMMAND python3
   tools/lint_timing_naming.py)` — single-line edit. CTest now fails
   on any new violation.

3. **Document the gate in the trigger tables.**
   - `.claude/CLAUDE.md` Documentation Sync trigger table gets a row:
     "New cross-stage accessor in the timing model" → "lint passes
     (enforced via CTest)".
   - `/resources/timing_discipline.md` Forbidden Patterns section
     notes that the four bullets added in Phase 0 are now CI-enforced.

**Phase 6 acceptance:**
- `python3 tools/lint_timing_naming.py` exits 0 on the post-Phase-5
  source.
- `ctest -R timing_naming_lint --output-on-failure` passes.
- Introducing a deliberate violation (e.g. renaming
  `BranchShadowTracker::current_in_flight` back to `is_in_flight`) and
  re-running CI fails the lint test — confirming the gate is live.

If a future change has a legitimate reason to deviate (e.g. a new
cycle discipline not yet covered by the two prefixes — there isn't a
known one, but the plan should leave the door open), the contributor
widens the prefix set in Phase 0's coding-standard section, the
timing-discipline doc, and the lint together. Per-edge-case
exemptions use the `// timing-naming-allow: <reason>` annotation that
silences a single lint finding; abuse of this annotation is a
code-review matter, not a lint matter.

## Critical files

To **read** (C++ side):

- `sim/include/gpu_sim/timing/pipeline_stage.h` — base hierarchy, no methods to rename but the header gets a docstring referencing the new convention.
- `sim/include/gpu_sim/timing/execution_unit.h` — second base hierarchy; `has_result()` rename in Phase 1 propagates here.
- All five execution-unit headers (`alu_unit.h`, `multiply_unit.h`, `divide_unit.h`, `tlookup_unit.h`, `ldst_unit.h`) — overrides of `has_result()`.
- `sim/include/gpu_sim/timing/scoreboard.h`, `branch_shadow_tracker.h`, `cache.h`, `coalescing_unit.h`, `decode_stage.h`, `load_gather_buffer.h`, `memory_interface.h`, `operand_collector.h`, `warp_scheduler.h`, `writeback_arbiter.h` — accessor renames (Phase 1) and field-shape changes (Phase 2).
- `sim/src/timing/timing_model.cpp:127–208` — wiring section updated for Phase 2 (`set_dependencies()` replaces `set_consumers()`); also the panic-flush cascade and lifetime ordering.
- `sim/src/timing/warp_scheduler.cpp:50–66` — `query_unit_ready` inline (Phase 3).
- `sim/include/gpu_sim/timing/redirect_request.h` and `sim/src/timing/fetch_stage.cpp:91–109` / `decode_stage.cpp:40–63` — `read_redirect_request` move (Phase 3).
- `tests/test_warp_scheduler.cpp`, `tests/test_timing_components.cpp`, `tests/test_branch.cpp`, `tests/test_panic.cpp` — fixture updates for Phase 1 renames and Phase 2 wiring change.

To **modify** (Phase 0 — documentation and lint):

- `/resources/cpp_coding_standard.md` — new section "Cross-stage accessor naming" (canonical home for the rules).
- `/resources/timing_discipline.md` — Forbidden Patterns expansion plus new "New module checklist" subsection; rows 1, 5, 8, 9, 11, 14, 15 also get accessor-name updates as Phase 1 lands.
- `/resources/onboarding.md` — Code Conventions section pointer.
- `.claude/CLAUDE.md` — General Instructions one-liner pointer; Documentation Sync trigger table updated in Phase 6 with the lint enforcement row.
- `tools/lint_timing_naming.py` — new file (Phase 0); flag flip in Phase 6.
- `CMakeLists.txt` (or `tests/CMakeLists.txt`) — `add_test(NAME timing_naming_lint COMMAND python3 …)` registration.

To **modify** (extractor side, after each C++ phase):

- `tools/diagram_extract_ast.py` — drop entries from `EDGE_CLASSIFICATION_OVERRIDES`, `EDGE_LABEL_OVERRIDES`, and `MANUAL_AST_EDGES` as the C++ side is updated to make them unnecessary.
- `tools/diagram_extract_md.py` — simplify `ROW_OVERRIDES` for rows whose prose now produces correct edges via the cross-product parser.
- `tests/test_signal_diagram.py` — pinned 10-edge floor doesn't change, but the equivalence test is the load-bearing assertion across this plan.

## Tricky parts

- **Polarity flips in Phase 1 are silent under the type system.** The compiler doesn't flag `if (foo->ready_out())` becoming `if (foo->current_busy())` — both are `bool`-returning. Every flipped-polarity rename has to be followed by a careful grep of the old name across the tree, plus a workload-benchmark cycle-count check after the commit. Cycle counts are the only mechanical proof that no flip was missed. Mitigation: do the rename and inversion in one atomic commit per accessor (don't batch), so a regression bisects to a single accessor.

- **`OperandCollector::ready_out()` reframes as `current_busy()` requires a polarity flip at every site.** Five sites in `WarpScheduler::evaluate` (one for opcoll, four for the unit ready cascade), one site in the panic-drained query. Same shape applies to each `ExecutionUnit::ready_out()`. Mitigation: rename + flip, then run benchmarks. If any benchmark drifts, bisect to the accessor that missed a site.

- **`WritebackArbiter::has_pending_work()` reframes as state, not possession.** Existing name suggests possession (`has_<noun>`); under the convention it's a state predicate (the arbiter holds pending state vs. is drained). Renaming to `current_busy()` matches the broader convention, but readers familiar with the old name will need to relearn. Document the change explicitly in the commit message and the timing_discipline.md row 14 update.

- **Phase 2 ripples through every test fixture.** `WarpScheduler`'s constructor signature changes. Catch2 fixtures that hand-roll a `WarpScheduler` need `set_dependencies()` calls. Mitigation: keep the old-signature constructor as a deprecated overload in `[[deprecated]]` mode for the duration of the rename, then delete after all callers are migrated.

- **`compute_ready` is gone, but the consolidation pass left some references.** Dead-code grep for `compute_ready` should come up empty before Phase 1 starts; if it doesn't, that's a sign the consolidation pass didn't reach everywhere.

- **The `is_stalled` → `next_stalled` rename reads slightly oddly.** The name `is_stalled` is widely used in trace output, benchmark prose, and review comments. The cycle-prefix rename loses the natural-English shape. Tradeoff is: the prefix mechanically encodes the COMBINATIONAL discipline at every call site, which the existing name does not. Accept the tradeoff; the prefix consistency is more valuable than the English fluency.

- **Diagram back-pressure styling becomes a derived property.** Pre-rename, the diagram extractor classified READY/STALL edges directly from the accessor name. Post-rename, the back-pressure direction is derived from cluster topology (`MODULE_CLUSTER` ordering) and an edge's `(src, dst)` cluster pair. The renderer's `EDGE_STYLE` table grows a small computed-direction step. If a future refactor re-orders clusters, back-pressure detection must be re-validated. Mitigation: the snapshot test pins a small set of known back-pressure edges with their expected render style.

- **Phase 4 is optional.** If touching the consolidation isn't paying off, skip it. It costs more in churn than it saves in extractor logic. Do it only when adjacent code is being modified for an unrelated reason.

- **TimingModel-level edges remain manual.** `DecodeStage::current_ebreak_request()` is read in `TimingModel::tick()` (top-of-tick observation), and `L1Cache` trace events are read in `TimingModel::record_cycle_trace()`. The panic-flush cascade calls `flush()` on four targets from `TimingModel::tick()` but the conceptual source is `PanicController`. None of these are fixable by the renames in this plan; they need either (a) walking `TimingModel::tick()` semantically or (b) accepting the `MANUAL_AST_EDGES` floor of ~6 entries.

## Verification

After each phase:

```
cmake -B build && cmake --build build -j8
cd build && ctest --output-on-failure
bash tests/run_workload_benchmarks.sh --build-dir build
python3 tools/render_signal_diagram.py --validate
python3 -m unittest tests.test_signal_diagram
python3 tools/lint_timing_naming.py    # Phase 0+; report-only until Phase 6
```

Acceptance gates per phase:

- All ctest cases green (including the registered `timing_naming_lint`
  test, which is report-only until Phase 6).
- Workload benchmark cycle counts byte-identical to the pre-phase
  baseline (use `python3 tools/bench_compare.py --baseline
  <pre-phase-sha>` to verify).
- `render_signal_diagram.py --validate` continues to report no
  differences (the AST and markdown extractors stay in sync as the
  C++ source changes; both should pick up the renames simultaneously).
- `tests.test_signal_diagram` passes — the 10-edge floor is invariant
  under naming changes, and the equivalence assertion catches drift.
- `lint_timing_naming.py` violation count strictly decreases each
  phase (Phase 0 captures a baseline; each later phase eliminates a
  documented subset; Phase 6 acceptance is zero violations).

End-state metrics:

- The signal taxonomy has two cycle disciplines (REGISTERED,
  COMBINATIONAL) and one orthogonal direction axis (forward-data,
  back-pressure). `timing_discipline.md` § Signal classifications and
  the per-boundary inventory both reflect this.
- Every cross-stage accessor matches one of three postfix shapes
  (state predicate / possession predicate / payload accessor) with
  asserted-blocking polarity for state predicates.
- `tools/diagram_extract_ast.py` `EDGE_CLASSIFICATION_OVERRIDES` empty, `EDGE_LABEL_OVERRIDES` empty, `MANUAL_AST_EDGES` reduced to the documented orchestrator-level floor (~6 entries: ebreak, trace events, panic-flush cascade). Pass D classification reduces to a one-line prefix check.
- `tools/diagram_extract_md.py` `ROW_OVERRIDES` reduced to rows where the prose genuinely needs disambiguation (5, 7, 13 are likely keepers; 8, 9, 14, 15 should disappear).
- `tools/render_signal_diagram.py` derives back-pressure styling from cluster topology rather than from accessor naming.
- `tools/lint_timing_naming.py` reports zero violations and is enforced as a CI gate covering prefix, postfix, polarity, and field-shape rules.
- New cross-stage accessors added in future PRs follow the convention by default — caught at PR review and as a hard CI failure if missed.

## Sequencing recommendation

**Phase 0 is mandatory and must land first** — every later phase
references its artifacts, and the lint provides the per-phase
violation-count gate that proves progress. No shortcut.

Phase 1 alone is worth the work — it's the single biggest source of
override entries and the rename is mechanical. Phases 2 and 3 are
medium-effort, high-value follow-ups. Phase 4 is optional. Phase 5 is a
hardening pass that's worth doing once but doesn't need to be its own
project.

**Phase 6 must land after Phase 5** — the lint can only flip to
enforcement once the source is clean. If Phases 4 or 5 are skipped,
Phase 6 still works, but the lint may need to grow per-line allow
annotations for the deferred items.

If only one phase ships beyond Phase 0: **Phase 1**. If two beyond
Phase 0: **Phase 1 + Phase 3**. The marginal value of Phase 2 is
mostly in extractor simplicity rather than C++ readability; do it
after Phase 3 has demonstrated the cross-stage-read-has-a-static-
call-site rule pays off. **Phase 6 should follow whichever C++ phases
ship**, not be deferred — partial enforcement is still enforcement,
and the lint scope can be narrowed to cover only what the completed
phases addressed (e.g., enable the `^current_*` REGISTERED check after
Phase 1 even if Phase 5's accessor enforcement is deferred).
