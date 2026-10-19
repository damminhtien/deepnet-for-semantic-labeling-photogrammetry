# Decisions

The first three entries are historical decisions from the completed workflow refactor. They do not describe the scope of the current code repair.

## 2026-10-19: Keep AGENTS.md as a routing table

Detailed architecture, data invariants, and long-running task procedures live under docs/ and .agent/. This keeps injected instructions short while preserving a clear path to the needed context.

## 2026-10-19: Keep this change documentation-only

The task is to refactor the agent workflow. Existing model, label, and notebook issues are recorded as constraints or follow-up risks; changing them would mix a separate behavior change into this task.

## 2026-10-19: Keep local Graphify output out of Git

The repository has no existing Graphify graph, and this task does not change code. Add /graphify-out/ to .gitignore; run graphify update . after future code changes as directed by AGENTS.md.

## Current task decisions

## 2026-10-19: Centralize data and evaluation contracts before model extraction

Move label conversion, input assembly/scaling, sampling, loss, metrics, dataset loading, and tiled inference into `semantic_labeling/`. Keep notebook model classes in place for this repair so the data/model inconsistencies can be fixed without changing the architecture boundary in the same step.

## 2026-10-19: Treat class ID 6 as an ignore sentinel

The six model output channels represent classes 0 through 5. Black no-boundary pixels map to ID 6 and must be ignored by both loss and evaluation; they are not a seventh predicted class.

## 2026-10-19: Preserve historical input scaling pending data evidence

Centralize the existing `/255` scaling so training and inference agree. Do not call it calibrated normalization: DSM is described as float32, and no reproducible run metadata or checkpoints are available to justify a new scale.

## 2026-10-19: Reject evaluation windows that leave image gaps

Require each sliding-window step to be no larger than its window size. Edge alignment alone cannot cover pixels when the stride exceeds the patch dimensions; rejecting that configuration prevents silent class-0 predictions in uncovered areas.

## 2026-10-19: Preserve model parameter names while fixing forward paths

Keep the six-class output and module names stable. In eval mode, PSPNet returns only main logits and skips auxiliary-head work; U-Net bilinear upsampling uses the configured feature-channel dimensions and rejects unknown methods.
