# Current state

## Task

Externalize the coding-agent workflow and keep AGENTS.md concise. See .agent/TASK.md.

## Status

Documentation and workflow refactor are complete. The validation results and task status are included in the workflow-refactor commit.

## Completed

- Inspected the initial Git state and Graphify setup.
- Ran graphify --help and graphify install --project --platform codex. A separate Graphify bootstrap commit is present: 9a305e1 (chore: configure project graphify support).
- Mapped the repository from the README, model scripts, dataset description, and evaluation code.
- Recorded the current input convention and the label-index discrepancy for follow-up.
- Added routing guidance, project architecture and invariants, and the fresh-thread workflow.
- Confirmed all eight task files are present, checked the AGENTS.md references and fresh-thread prompt, and kept AGENTS.md to 27 lines.
- Staged only the workflow-refactor files and confirmed git diff --cached --check passes.

## Current

- No active implementation work remains for this task.

## Next

- For a new task, replace TASK.md and PLAN.md, write its current progress to STATE.md, and start a task-specific decision log.

## Known issues

- test_kernel/test.py names label index 0 roads, while its palette names index 0 Impervious surfaces; the palette also includes index 6 while N_CLASSES is 6. This workflow task records the discrepancy but does not change the mapping.
- No root-level dependency manifest or test configuration was found during this audit. Do not assume a standard Python test command.

## Validation

- Documentation paths and the fresh-thread prompt: passed.
- git diff --cached --check: passed.
- Runtime/model tests: not run; this change does not modify code, notebooks, or data.
- graphify update .: not run; this change does not modify code and no Graphify graph exists yet.
