# Goal

Fix root causes behind verified data, label, evaluation, and model-forward inconsistencies. Create a shared Python implementation for repeated pipeline utilities and update repository notebooks/scripts to use it, leaving a safer base for the larger refactor.

# Constraints

- Preserve the six model output classes and existing checkpoint parameter names.
- Keep notebook source edits on Jupyter MCP; do not edit `.ipynb` files through filesystem tools.
- Do not run training or benchmark workloads; the checkout has no dataset, checkpoint, or project Python environment.
- Preserve historical experiment outputs and unrelated local files.
- Record unresolved assumptions instead of inventing dataset normalization or split rules.

# Acceptance

- Shared modules own label conversion, sampling, loss, metrics, input normalization, and patch loading.
- Training/evaluation notebooks and `test_kernel/test.py` use the same helpers.
- Verified model forward-path bugs are fixed in source and notebook copies.
- Static syntax checks and `git diff --check` pass; Graphify is updated.
- `.agent/STATE.md` records exact validation and unresolved limitations.
