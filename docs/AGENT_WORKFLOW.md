# Long-running coding-agent workflow

The repository is the durable record for a task. Keep the goal, plan, observed progress, decisions, and next action in files so a new Codex thread can resume without replaying the conversation.

## State files

- .agent/TASK.md freezes the current task goal, constraints, and acceptance criteria.
- .agent/PLAN.md splits multi-file work into milestones that can each be implemented and verified independently.
- .agent/STATE.md records completed work, current work, the exact next action, known issues, validation results, and the latest checkpoint.
- .agent/DECISIONS.md records decisions for the active task, including the reason and any consequence.

Keep durable repository rules in AGENTS.md or docs/. Keep task-specific state in .agent/. When a new task starts, replace the active task and plan, carry forward only unresolved state, and start a new decision log. Git history preserves completed task records.

## Verification boundaries

Split work by independently verifiable outcomes, not by arbitrary amounts of coding. A milestone should state:

1. the behavior or artifact that will exist when it is done;
2. the smallest relevant validation command or observable result;
3. any dependency on earlier milestones.

For each milestone, implement, validate, inspect the result, update PLAN.md and STATE.md, then create a commit or another recoverable checkpoint when appropriate. Repair failed validation before moving on. Keep acceptance criteria specific to the task; do not copy generic commands such as pytest or mypy unless this repository and the changed code support them.

## Context handoff

After two or more compactions, or when the thread repeatedly rereads the same files without closing a milestone, checkpoint the work and start a fresh thread. Update STATE.md with completed work, current implementation status, important decisions, unresolved issues, validation commands and results, the checkpoint, and the exact next step.

Use this prompt for the new thread:

    Read AGENTS.md, .agent/TASK.md and .agent/STATE.md.

    Continue the current task from STATE.md.
    Inspect only additional files required for the next unfinished item.

Read PLAN.md and DECISIONS.md only as needed for the next unfinished item. Do not treat a successful local check as evidence of an experiment, deployment, or benchmark result unless that result was actually observed.
