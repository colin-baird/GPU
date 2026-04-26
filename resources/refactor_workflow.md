# Refactor Workflow

Workflow for **behavior-preserving multi-phase refactors** — changes that
restructure code without changing observable behavior. The canonical example is
the Phase 1–8 cross-stage signaling refactor described in
[`/resources/timing_discipline.md`](timing_discipline.md), followed by the
consolidation pass that landed once Phase 8 shipped.

This is distinct from the
[`multi-agent-workflow`](/.claude/skills/multi-agent-workflow/SKILL.md) skill,
which is shaped for adding architectural features (implement → validate →
test-author with an adversarial fix loop). For refactors that shape is wrong:
there is no new behavior to test-author against — the existing regression suite
*is* the contract.

## When to use

Use this workflow when **all** of the following are true:

- The change restructures code, renames concepts, eliminates duplication, or
  consolidates abstractions.
- It is broken into 3+ phases that ship over multiple commits.
- The contract is "behavior preserved" — typically cycle counts byte-identical
  to baseline, or with a documented, justified delta listed in the phasing doc.

Do NOT use this workflow when:

- The change introduces new architectural behavior or changes a spec claim →
  use [`multi-agent-workflow`](/.claude/skills/multi-agent-workflow/SKILL.md).
- A measurable performance delta is the goal → use `multi-agent-workflow`; its
  benchmark-comparison gate is what you want.
- The change ships in a single commit and fits in one agent's context → just
  do it directly. Neither workflow is needed.

## Phases

### 1. Plan

Write `project-plans/<refactor-name>.md` decomposing the work into phases. Each
phase should be small enough to be locally correct without revisiting earlier
phases. Capture the regression invariant explicitly:

- "Cycle counts byte-identical to baseline", OR
- A list of documented deltas with justification.

### 2. Baseline

Before phase 1 lands, capture a baseline benchmark snapshot:

```
python3 tools/bench_compare.py --baseline HEAD
```

The `RESULT name=X cycles=Y` lines are what every phase's validation compares
against. Note the baseline commit SHA in the phasing doc.

### 3. Per-phase loop

For each phase:

1. **Implement.** Orchestrator directly, or via the implement agent. The phase
   brief should include: scope of THIS phase only, the regression invariant,
   and a pointer to the phasing doc so the implementer understands where the
   phase fits.
2. **Validate.** Build + regression + benchmark delta:
   - `cmake -B build && cmake --build build -j8` (must succeed)
   - `ctest -j8` from `build/` (all tests pass)
   - `python3 tools/bench_compare.py --baseline <baseline-sha>` (deltas zero
     or matching the documented list)
3. **Commit.** One phase = one commit. Bundle implementation + spec updates
   per the standard CLAUDE.md commit rule.

### 4. Consolidation review

Run **every ~3 phases**, AND **before the final commit of the refactor**.
Dispatch the consolidation-reviewer agent
(`.claude/agents/consolidation-reviewer.md`) with the diff range:

```
Review diffs from <phase-1-parent>..HEAD. Report duplication, dead code,
and accumulated cruft per your checklist. Report only — do not fix.
```

The agent reads the range with `git log -p` / `git diff` and reports findings
in three buckets: drift risk, dead code, API sprawl. Orchestrator decides
scope of follow-up.

Apply consolidation as 1+ cleanup commits (regression-gated like every other
phase). **Skip test-authoring** — regression is the contract.

### 5. Doc sync

Walk the trigger table in CLAUDE.md as for any other change. For refactors
that change file responsibilities or remove abstractions, `perf_sim_arch.md`
is the most commonly affected. The phasing doc and `timing_discipline.md`
(if applicable) should also be updated to reflect what landed.

## What's different from multi-agent-workflow

- **No test-authoring round.** Regression is the contract; the existing tests
  are the adversary.
- **Stricter validation:** cycle-count byte-equivalence (or documented delta)
  vs. just "tests pass."
- **Consolidation review is a first-class phase**, not optional.
- **Per-phase commits are encouraged.** Each phase is self-contained.
- **No adversarial fix loop.** If regression breaks, the phase is wrong —
  fix or revert; do not test-author around it.
- **No `UNTESTED.md` entry** for the refactor itself (no new behavior to
  test). The existing tests cover what the refactor preserves.

## Worked example

The Phase 1–8 cross-stage signaling refactor (commits `f4f1cb5..bbb1de3`)
followed by the Consolidation pass (`54e6542`). The phased work shipped
phase-by-phase. A single consolidation review at the end surfaced ~20 hygiene
issues (predicate triplication, dead `current_*` mirrors, `flush()` ↔
`reset()` drift, override-field sprawl, stale "for tests/tooling" methods)
that no individual phase would have flagged on its own. Net: 35 files,
+386 / −571, zero cycle-count change across all six benchmarks.
