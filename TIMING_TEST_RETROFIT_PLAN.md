# Timing Test Retrofit Plan

Working plan for auditing and augmenting `sim/tests/` so existing tests exercise the timing behavior they claim to cover, instead of being oracle-carried by the functional model. Delete this file when the retrofit is complete.

## Background

The functional model in this simulator acts as an oracle: it pre-populates register and memory results before the timing model exposes them. This makes it easy to write tests that pass even when the timing code is broken — a test that asserts "the right value eventually appears in a register" is trivially satisfied regardless of cycle behavior.

A review of recent test additions (MSHR merging, branch shadow, TLOOKUP pipelining, decode-buffer depth, gather buffer) found:

- Strong component-level state-machine coverage.
- Weak negative-space timing coverage (tests rarely check that an event does *not* happen in earlier cycles).
- Arbitration edges underspecified — priority rules like `FILL > secondary drain > HIT` usually have only one of the three pairwise edges tested.
- Pipelining claims ("64 → 17 cycle latency, pipelined dual-port BRAM") tested for latency only, never for initiation interval.
- Performance-motivated commits ("stall while in branch shadow", "depth 3 for shadow tolerance") with tests that would still pass if the new gate were removed.

## Agent prompt updates already made (prerequisites)

These are the contract updates the retrofit builds on. A fresh context window should re-read them before starting.

- [`.claude/agents/test-author.md`](.claude/agents/test-author.md) — new "Timing is the hard part" section. Names the oracle problem, defines the self-check ("would this test pass if timing were stubbed to 0 cycles?"), catalogs the five timing-claim categories, and gives default test shapes (negative-space latency, back-to-back pipelining, eligibility gate, arbitration contention).
- [`.claude/skills/multi-agent-workflow/SKILL.md`](.claude/skills/multi-agent-workflow/SKILL.md) — new "Timing-claim enumeration" subsection requiring the orchestrator to produce an explicit manifest (claim / spec-section / file-function) before dispatching test-author.

## Strategy

**Supervisor gathers, sub-agent writes.** The orchestrator extracts timing claims from the spec and cross-references existing tests — this is the expensive, context-heavy work. The test-author sub-agent consumes that manifest and writes/rewrites tests mechanically.

**Triage before dispatching.** Not every test file carries timing debt. A one-pass inventory prevents burning sub-agent dispatches on low-debt components.

**Per-component loop, not one big dispatch.** A single sub-agent holding 8 components of context produces shallow work on the later ones. Per-component gives clean commits, easy revert granularity, and an abort point if returns diminish.

**Structured manifest, not free-form "expected behavior."** The five-category schema from `SKILL.md` (latency / throughput / priority / ordering / stall) is the hand-off contract. If the manifest has 7 claims and the returned tests cover 5, the gap is visible.

## Phase 0 — Triage (orchestrator, no sub-agents)

Produce a single table covering every `sim/tests/test_*.cpp`:

| File | Timing claims in scope | Current timing coverage | Debt |
|------|------------------------|-------------------------|------|

Debt levels:
- **High** — timing claims with no coverage or only oracle-carried tests.
- **Medium** — partial coverage; missing edges, pipelining, or negative-space assertions.
- **Low** — file is not about timing (decoder encoding, ISA compliance, etc.).

Output: a ranked worklist. Stop partway if returns diminish.

## Phase 1 — Per-component retrofit loop

Suggested ordering by debt (from the review). Adjust after Phase 0 confirms.

1. **`test_branch.cpp`** — branch-shadow stall and fetch-block-on-near-full-decode-buffer have no direct timing tests. Highest-debt target.
2. **TLOOKUP cases in `test_timing_components.cpp`** — pipelining claim untested; four redundant latency tests to consolidate.
3. **`test_cache_mshr_merging.cpp`** — 2/3 arbitration edges missing; write-buffer path bypasses `evaluate()` in one case.
4. **Decode-buffer depth-3 cases in `test_timing_components.cpp`** — architectural motivation (fetch-bubble tolerance) untested; only container plumbing verified.
5. **Writeback arbiter cases in `test_timing_components.cpp`** — priority structure by definition; audit for contention tests.
6. **LD/ST, divide, multiply/VDOT8 cases in `test_timing_components.cpp`** — audit pipelining and back-to-back throughput claims.
7. **Scoreboard / issue-stage eligibility** — likely scattered across integration tests; may need a dedicated file.
8. **`test_load_gather_buffer.cpp`** — lower expected debt; verify and close.

### Per-component loop steps

1. **Orchestrator gathers.** Read relevant spec sections, implementation files, and the existing test file. Produce:
   - **Timing-claim manifest** — one row per claim: `claim (one sentence) | spec section | file/function where logic lives`. Use the five categories from `SKILL.md`.
   - **Coverage-gap list** — per claim: is there a test? Does it negative-space assert? Is it oracle-carried? Rewrite or augment?

2. **Dispatch test-author once per component** with:
   - The timing-claim manifest.
   - The coverage-gap list.
   - Instruction to *augment* the existing file, not replace it. Keep functional-correctness tests that are genuinely correctness-scoped. Rewrite oracle-carried "timing" tests into negative-space form. Consolidate redundant latency tests.
   - Path to the existing file for in-place edit.

3. **Validate.** Build and run regression for that file's targets.
   - New tests failing against current impl → classify: test-wrong or real-bug? If the fix would need a spec change, escalate per the adversarial-test-failure loop in `SKILL.md`.

4. **Commit per component.** One commit per component. Message: `Retrofit timing tests: <component> (N added, M rewritten)`. Small reviews; clean revert granularity.

5. **Update `UNTESTED.md`.** Remove entries the new tests cover. Add entries for timing claims that remain untested — sometimes a claim is too whole-pipeline-coupled to unit-test cleanly and belongs in `test_integration.cpp` instead.

## Pilot recommendation

Before committing to 8 iterations, run the full loop end-to-end on **`test_branch.cpp`** first. It's the highest-debt target and the smallest file. If the manifest → dispatch → validate → commit loop works there, it scales to the rest. If it doesn't, adjust the prompt contract before burning more dispatches.

## Deliverables

- [ ] Phase 0 triage table
- [ ] Pilot: `test_branch.cpp` retrofit committed
- [ ] Per-component retrofits committed (items 2–8 above)
- [ ] `UNTESTED.md` updated
- [ ] Regression suite still green at end
- [ ] Optional: `resources/test_timing_playbook.md` capturing patterns that emerged, so future changes don't re-discover them
- [ ] Delete this plan file

## Notes for the fresh context window

- The user is `cb3663@princeton.edu` (Colin Baird). Project is the FPGA GPU accelerator — see `/AGENTS.md` (symlink to `.claude/CLAUDE.md`).
- The user originally flagged that "timing alignment with what is expected given the spec is key since correctness issues don't usually surface since the functional simulator is acting as an oracle." That sentence is the north star for this work.
- The user approved this plan's strategy with the refinement that the orchestrator gathers expected behavior and delegates to test-author per-structure. Triage-before-dispatch and structured manifest were orchestrator additions to that strategy.
- Don't start Phase 1 without confirming with the user which pilot target to run first.
