# Decisions

## 2026-10-19: Keep AGENTS.md as a routing table

Detailed architecture, data invariants, and long-running task procedures live under docs/ and .agent/. This keeps injected instructions short while preserving a clear path to the needed context.

## 2026-10-19: Keep this change documentation-only

The task is to refactor the agent workflow. Existing model, label, and notebook issues are recorded as constraints or follow-up risks; changing them would mix a separate behavior change into this task.

## 2026-10-19: Keep local Graphify output out of Git

The repository has no existing Graphify graph, and this task does not change code. Add /graphify-out/ to .gitignore; run graphify update . after future code changes as directed by AGENTS.md.
