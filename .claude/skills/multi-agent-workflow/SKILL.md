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

Sub-agents run sequentially; each phase must complete before the next begins.

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
- Create all commits — sub-agents do not commit. Bundle implementation, spec, and tests in one atomic commit.
- Before committing, verify documentation per the Documentation Sync rules in `CLAUDE.md` (the trigger table there is authoritative and applies to both workflow and single-agent changes).
