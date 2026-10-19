# Goal

Refactor the repository's coding-agent workflow into compact routing guidance and durable task state that supports long-running work and fresh-thread continuation.

# Constraints

- Keep the workflow-refactor changes limited to documentation and agent state; the Graphify bootstrap is a separate commit.
- Do not modify model code, experiment notebooks, dataset files, or historical results.
- Record repository facts only when supported by files in this checkout.
- Preserve the installed .codex/ tooling; do not alter it in the workflow-refactor commit.

# Acceptance

- AGENTS.md is a short routing table with only high-value invariants.
- .agent/PLAN.md, .agent/STATE.md, and .agent/DECISIONS.md define verifiable milestones, current progress, and task decisions.
- docs/architecture.md, docs/invariants.md, and docs/AGENT_WORKFLOW.md carry detailed project and workflow guidance.
- The fresh-thread prompt is recorded verbatim and points to the current task state.
- Documentation diff passes git diff --check; no runtime or model claims are inferred from this documentation-only change.
