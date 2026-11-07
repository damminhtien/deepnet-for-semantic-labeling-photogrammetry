# Long-running coding-agent workflow

Record the task and its next step in `.agent/` so another chat can resume it.

## State files

- .agent/TASK.md freezes the current task goal, constraints, and acceptance criteria.
- .agent/PLAN.md splits multi-file work into milestones that can each be implemented and verified independently.
- .agent/STATE.md records completed work, current work, the exact next action, known issues, validation results, and the latest checkpoint.
- .agent/DECISIONS.md records decisions for the active task, including the reason and any consequence.

Keep repository rules in `AGENTS.md` or `docs/`. For a new task, start a plan
and decision log and carry forward unresolved state. Completed tasks are in Git
history.

## Verification boundaries

Give each milestone:

1. the behavior or artifact that will exist when it is done;
2. the smallest relevant validation command or observable result;
3. any dependency on earlier milestones.

After verifying each milestone, update `PLAN.md` and `STATE.md` and checkpoint
the work when appropriate. Fix failures before continuing.

## Context handoff

After two or more compactions, or when the thread repeatedly rereads the same files without closing a milestone, checkpoint the work and start a fresh thread. Update STATE.md with completed work, current implementation status, important decisions, unresolved issues, validation commands and results, the checkpoint, and the exact next step.

Use this prompt for the new thread:

    Read AGENTS.md, .agent/TASK.md and .agent/STATE.md.

    Continue the current task from STATE.md.
    Inspect only additional files required for the next unfinished item.

Read `PLAN.md` and `DECISIONS.md` as needed for the next unfinished item.

## Google authentication boundaries

Use the connected Google Drive integration to list, upload, download and verify
Drive files. Drive OAuth and gcloud Application Default Credentials (ADC) have
separate scopes.

Use ADC for Google Cloud APIs that accept its scopes. If the Drive API returns
`403 ACCESS_TOKEN_SCOPE_INSUFFICIENT` with ADC, switch to the Drive integration
and verify the files there. Reuse the authorized connection and keep credential
contents private.

To mount Drive in a Colab VM, call `google.colab.drive.mount()` in that session.

## Colab CLI authentication

The Colab CLI, website and VM have separate authentication. Use
`/Users/macbook/.local/bin/colab`, which defaults to gcloud ADC unless given
`--auth`. On machines without the wrapper, pass `--auth=adc` before the
subcommand to use ADC. An empty `colab sessions` response means there are no
active CLI-listed sessions for that account.

The upstream CLI writes request headers to `~/.config/colab-cli/colab.log` at
DEBUG level. The local wrapper redacts authorization values before logging. Use
the wrapper and keep raw debug logs private.
