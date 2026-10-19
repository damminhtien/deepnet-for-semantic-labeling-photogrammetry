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

## Google authentication boundaries

Use the connected Google Drive integration for Drive file operations such as listing, uploading, downloading, and verifying files. Do not use gcloud Application Default Credentials (ADC) as a substitute for the Drive connection: ADC and Drive OAuth are separate credentials, and a successful Google login does not imply that an ADC token contains Drive API scopes.

Use gcloud ADC only for Google Cloud APIs that accept the scopes on that ADC token. If a Drive API request made with ADC returns `403 ACCESS_TOKEN_SCOPE_INSUFFICIENT`, treat it as a credential-scope mismatch in the selected tool path, not as evidence that the user failed to log in. Switch to the connected Drive integration and verify the resulting files there. Do not ask the user to repeat OAuth when the Drive integration is already authorized, and never inspect or print credential contents.

Mounting Drive inside Colab is a separate runtime action: use `google.colab.drive.mount()` in that Colab session when it needs access. Do not route that step through local gcloud ADC.

## Colab CLI authentication

The Colab CLI's API authentication is separate from the Colab website session and from credentials injected into a running VM by `colab auth`. On this Mac, `/Users/macbook/.local/bin/colab` defaults commands to the existing gcloud ADC unless `--auth` is explicitly supplied. Use this wrapper rather than invoking the Python 3.14 entry point directly. On another machine without this wrapper, pass `--auth=adc` before the subcommand when using ADC. A successful `colab sessions` request with an empty assignment list means authentication succeeded but the account has no active CLI-listed sessions; it does not mean the browser notebook is signed in or running.

The installed upstream CLI writes DEBUG request headers to `~/.config/colab-cli/colab.log`, including authorization headers. The local wrapper redacts bearer and token-like values before log records reach any handler. Keep using the wrapper, and do not print credential contents or raw debug logs.
