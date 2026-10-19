# Engineering rules

- Keep changes focused and code readable.
- Add regression coverage for behavior changes where the repository has a suitable test path.
- Validate at the narrowest relevant boundary and report only checks that actually ran.
- Treat model scripts, experiment notebooks, and evaluation code as related; inspect relevant counterparts before changing model behavior.

# Repository map

- Architecture and file roles: [docs/architecture.md](docs/architecture.md)
- Data and experiment constraints: [docs/invariants.md](docs/invariants.md)
- Long-running task state: .agent/TASK.md and .agent/STATE.md
- Milestones and resume procedure: .agent/PLAN.md and [docs/AGENT_WORKFLOW.md](docs/AGENT_WORKFLOW.md)
- Current task decisions: .agent/DECISIONS.md

# Task workflow

- Start or resume from AGENTS.md, .agent/TASK.md, then .agent/STATE.md.
- For multi-file work, make .agent/PLAN.md milestones independently verifiable.
- Update .agent/STATE.md after each milestone and before a context handoff.

# Graphify

- If the user invokes /graphify, follow .codex/skills/graphify/SKILL.md.
- If graphify-out/graph.json exists, use graphify query for codebase exploration and graphify path or explain for focused relationships.
- After code changes, run graphify update .
- Keep generated graphify-out/ local; it is ignored by Git.
