---
name: multi-agent-workflow
description: Orchestrate the multi-agent architectural-change workflow for this project (implement → validate → test-author, with regression hard gate, adversarial fix loop, and commit integration). Invoke ONLY when the user explicitly asks to use the multi-agent workflow, orchestrator, or "the full workflow" for a change. Do not invoke for ordinary single-agent tasks.
---

# Multi-Agent Architectural Change Workflow

This skill defines the orchestration state machine the main conversation agent (the orchestrator) follows when the user has explicitly requested the multi-agent workflow for a change.

Sub-agent prompts:
- Implementation: `.claude/agents/implement.md` — writes code + spec updates.
- Validation: `.claude/agents/validate.md` — builds, runs regression, runs benchmarks. Does not modify code.
- Test Authoring: `.claude/agents/test-author.md` — writes adversarial Catch2 tests. Does not modify implementation.
- Consolidation Review: `.claude/agents/consolidation-reviewer.md` — reads recent diffs and reports duplication, dead code, and accumulated cruft. Does not modify code.

Sub-agents run sequentially; each phase must complete before the next begins.

> **For behavior-preserving multi-phase refactors** (no new behavior, regression-as-contract), use the lighter `resources/refactor_workflow.md` checklist instead of this skill. It skips test-authoring and the adversarial fix loop, and makes consolidation review a first-class phase.

## Architectural Change Workflow

```
User requests change (with multi-agent workflow)
        │
        ▼
┌─────────────────┐
│  Implementation │  Writes code + spec update
└────────┬────────┘
         │ build must succeed
         ▼
┌─────────────────┐
│  Validation     │  Runs regression suite
└────────┬────────┘
         │ HARD GATE: all regressions must pass
         │ If failures → fix or revert, do not proceed
         ▼
┌─────────────────┐
│  Validation     │  A/B benchmark comparison vs. baseline
└────────┬────────┘
         │
         ▼
    ┌────────────┐
    │ Major perf │── Yes ──→ Test Authoring writes targeted tests
    │ win?       │           → tests pass: commit impl + spec + tests
    └────┬───────┘           → tests fail: Adversarial Test Failure loop
         │ No / Unclear
         ▼
    Orchestrator consults user with regression + benchmark data
         │
    ┌────┴────┐
    │ Keep?   │── Yes ──→ Test Authoring writes tests → commit together
    └────┬────┘           (failures → Adversarial Test Failure loop)
         │ No / No response within session
         ▼
    Revert or stash the change
```

## Consolidation Review (before final commit)

Before the final commit of any change that has either (a) accumulated across
2+ iterations of the adversarial fix loop, (b) ships ≥ 200 lines of net diff,
or (c) touches ≥ 5 files, dispatch the consolidation-reviewer agent
(`.claude/agents/consolidation-reviewer.md`) with the diff range from the
change's starting point to `HEAD`.

The agent reads the range and reports duplication, dead code, override-field
sprawl, and similar accumulated cruft. It does NOT fix issues — the
orchestrator decides whether to apply consolidation as part of the final
commit, defer it to a follow-up commit in the same session, or ignore the
findings.

**Why it matters:** the implement → fix-loop → fix-loop sequence tends to
layer new abstractions on top of old ones. Each iteration is locally correct
but no one looks back. The reviewer's pre-commit pass catches the residue.

For small single-iteration changes (one fix, no loop, < 200 line diff),
skip this step.

## Adversarial Test Failure Loop

When adversarial tests expose a bug:

1. **Classify the bug.**
   - **Escalate immediately** if the fix would require changing the architectural spec, redesigning an interface, changing pipeline semantics, or the root cause is ambiguous after reading the code.
   - **Enter fix loop** if self-contained: wrong constant, missed switch case, bit-field extraction error, logic bug in a single function.

2. **Fix loop (max 2 iterations):**
   - Implementation agent — bug fix only, no refactor or scope expansion.
   - Validation agent — full regression + new adversarial tests.
   - If pass: commit impl + spec + tests together.
   - If fail after 2 iterations: escalate to user with both failing tests, both attempts, and a diagnosis. Do not attempt a third fix autonomously.

## Decision Criteria

- **Major performance win**: measurable IPC improvement on a representative workload, or reduction in a critical-path latency. Evaluated from `tools/bench_compare.py` output.
- **Regression hard gate**: any regression failure stops the workflow. Benchmarking does not proceed with broken regressions.
- **Session timeout**: if the user is consulted and does not respond within the session, default to shelve (revert/stash).
- **Fix loop scope**: the implementation agent may only modify the code path identified as buggy — no refactor, no new features.

## Untested Change Logging

If a change is kept but targeted tests are deferred, log it in `/UNTESTED.md`. Remove the entry when tests are committed or the change reverted.

## Orchestrator Responsibilities (during workflow)

- Dispatch sub-agents in the correct sequence; enforce hard gates.
- Classify bugs before entering the fix loop.
- Consult the user at every decision sub-agents cannot resolve.
- Never proceed past a failed regression gate or past the fix-loop limit without user input.
- Run consolidation review before the final commit when the trigger conditions above apply.
- Create all commits — sub-agents do not commit. Bundle implementation, spec, and tests in one atomic commit.
- Before committing, verify documentation per the Documentation Sync rules in `CLAUDE.md` (the trigger table there is authoritative and applies to both workflow and single-agent changes).

### Timing-claim enumeration (hand-off to test-author)

Before dispatching the test-author, read the implementation diff and the spec diff and produce an explicit enumeration of the **timing claims** the change makes or modifies. Include this enumeration verbatim in the test-author prompt. This is how the orchestrator prevents timing regressions from slipping past the oracle.

Enumerate every:

- **Latency change** — e.g., "TLOOKUP: 64 → 17 cycle latency"
- **Throughput / pipelining claim** — e.g., "TLOOKUP is pipelined dual-port BRAM; back-to-back accepts allowed with II < 17"
- **Priority / arbitration rule added or changed** — e.g., "cache gather-extract port: FILL > secondary drain > HIT"
- **Ordering / sequencing rule** — e.g., "chain drains in allocation order; pin clears only when the last secondary retires"
- **Stall / eligibility gate added or changed** — e.g., "warp ineligible while `branch_in_flight` set", "fetch blocked when decode buffer is one slot from full"
- **Counter / stall-reason whose increment timing is itself a spec claim** — e.g., "`line_pin_stall_cycles` increments once per cycle a different-tag miss is rejected while the set is pinned"

For each entry, state: (a) the claim in one sentence, (b) the spec section or line(s) it lives in, (c) the file/function where the timing logic lives. The test-author uses this list as its primary test target — a missing entry will usually mean a missing test.

If the change makes no timing claims (pure functional bug fix, doc change, refactor with identical cycle behavior), say so explicitly. Do not pass an empty list implicitly.
