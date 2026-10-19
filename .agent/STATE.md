# Current state

## Task

Fix root causes behind the verified data, label, evaluation, and model-forward inconsistencies, then leave a clean base for the larger architecture refactor. See `.agent/TASK.md`.

## Status

Implementation, static validation, diff review, and the task-owned Git checkpoint are complete.

## Completed

- Audited the evaluation script, five training notebooks, test notebook, data-inspection notebook, three model-visualization notebooks, and matching Python model scripts.
- Added `semantic_labeling/` as the shared source for the six-class map, label conversion, five-channel assembly/scaling, patch/window sampling, ignore-aware loss and metrics, dataset loading, tiled inference, and model initialization.
- Migrated training/evaluation callers and notebook copies to the shared contracts. Replaced duplicated label/data/evaluation helpers and removed unsafe import-time model execution from `test_kernel/test.py`.
- Fixed the label mismatch: black is `IGNORE_LABEL = 6`, excluded from the six output classes by loss and metrics.
- Fixed PSPNet evaluation so it does not access auxiliary logits in eval mode, repaired U-Net bilinear channel configuration, and replaced hardcoded five-channel model dimensions with `INPUT_CHANNELS` across source and notebook copies.
- Made invalid sliding-window strides fail instead of leaving pixels unscored; made all-ignore batches produce a differentiable zero loss instead of NaN.
- Kept the historical `/255` convention centralized and documented that it is not a calibrated normalization for float32 DSM values.
- Updated architecture/invariant documentation and recorded this task's decisions.

## Current

- Shared input, label, sampling, dataset, loss, metric, and inference behavior now has one implementation. Model architecture definitions still exist in both notebooks and scripts; moving them into importable modules is a separate next refactor.
- Only task-owned tracked paths should be staged. Preserve `.jupyter/` and `.jupyter_ystore.db`; they are local Jupyter runtime state. Ignore and do not stage Graphify output or notebook checkpoint directories.

## Next

- For the larger refactor, move one model architecture at a time from notebook copies into importable modules with an independently verifiable boundary.

## Known issues and limits

- The checkout has no project dependency manifest, dataset, model checkpoint, or installed NumPy/PyTorch/scikit-learn environment. Training, inference, and benchmark runtime behavior therefore remain unverified.
- Uniform `/255` scaling is retained to preserve the historical pipeline; DSM is described as float32, so physical normalization still needs a separate evidence-backed decision.
- Notebook model architectures remain duplicated and require a future extraction. External notebook runtimes must have this repository on their Python path.
- The README's historical accuracy figures have no matching run manifest or checkpoint in this checkout.

## Validation

- `rtk ruff check semantic_labeling model_script test_kernel/test.py`: passed.
- `python3 -m compileall -q semantic_labeling model_script test_kernel/test.py`: passed.
- Jupyter MCP `compile()` syntax pass over 43 changed code cells: passed; notebook training/inference cells were not executed.
- `graphify update .`: passed; rebuilt 169 nodes, 260 edges, and 14 communities.
- `git diff --check` and `git diff --cached --check`: passed after final edits and staging.
- Model runtime, training, and benchmark checks: not run because dependencies, data, and checkpoints are absent.
