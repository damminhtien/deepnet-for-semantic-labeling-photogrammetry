# Plan

## M1: Audit the existing repository

Status: complete

- Inspect the initial Git state, repository instructions, README, model scripts, and evaluation code.
- Identify existing workflow files and preserve unrelated local state.
- Completion check: architecture notes are limited to facts verified in the checkout.

## M2: Externalize workflow and project guidance

Status: complete

- Keep AGENTS.md focused on engineering rules, routing, and key invariants.
- Add task, plan, state, and decision files under .agent/.
- Add architecture, data invariants, and resume workflow under docs/.
- Completion check: the continuation order and fresh-thread prompt are explicit.

## M3: Verify and checkpoint

Status: complete

- Check changed-file scope, local Markdown references, and git diff --cached --check.
- Verify the continuation prompt and keep AGENTS.md under 35 lines.
- Do not run model or notebook workloads for this documentation-only task.
- Stage only task-owned documentation and commit it; leave the installed .codex/ configuration unchanged.
- Completion check: validation results and final task status are recorded in STATE.md.
