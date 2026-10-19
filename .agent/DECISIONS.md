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

## 2026-10-19: Start a new PSPNet experiment from random initialization

The user chose training from scratch. Initialize the ResNet-101 trunk and PSPNet heads randomly, with no historical PSPNet checkpoint or ImageNet weights. Use 18 labeled scenes for training, the notebook's six listed scenes for validation, and the remaining 14 scenes for held-out evaluation. Preserve `/255` scaling, six output classes, ignore label 6, and the original Adam loss setup.

## 2026-10-19: Keep PSPNet pooling on MPS

MPS adaptive average pooling rejects some non-divisible feature/bin sizes used by PSPNet. Use explicit separable mean reductions with the same floor/ceil bin boundaries, and verify output parity against CPU adaptive pooling before training.

## 2026-10-19: Use the Mac for data work and two independent cloud replicas

The user paused local training after epoch 2 and chose Colab and Kaggle for parallel runs. Both jobs resume from the same saved MPS model and Adam checkpoint, but maintain separate optimizers, randomness, checkpoints, and metrics. Do not describe them as distributed training. Keep the Kaggle dataset private and use manual OAuth; never include credential clippings or auth material in the bundle.

## 2026-10-19: Preserve input semantics while reducing cache memory

Cache raw aligned image rasters and normalize only sampled patches. A real patch matched the former full-tile `/255` pipeline exactly. Keep five-channel `/255` behavior unchanged and record the observed DSM range.

## 2026-10-19: Restrict near-palette decoding to Potsdam no-boundary tile 4_12

All train labels and the other evaluation references use exact RGB palette decoding. The no-boundary `4_12` tile has systematic color drift; accept nearest-palette mapping only for this tile, enforce unique nearest colors and a maximum channel error of 64, and record pixel count and observed error. Do not add validation/test class distributions to preprocessing summaries.

## 2026-10-19: Give Colab and Kaggle different model jobs

Colab is the only PSPNet continuation, resuming the saved epoch-2 model through epoch 16. Kaggle runs a separate supervised SAM ViT-B experiment; do not upload the PSPNet checkpoint to its kernel or train PSPNet there. The Mac remains dedicated to data analysis, preprocessing, augmentation, and postprocessing.

## 2026-10-19: Fine-tune SAM ViT-B with LoRA and a semantic decoder

Initialize the Kaggle SAM branch from Meta's official ViT-B checkpoint, freeze its original image encoder weights, and train rank-4 LoRA updates on each attention qkv/proj projection together with a custom decoder for classes 0–5. Ignore label 6 in training and evaluation. The decoder predicts semantic class maps; SAM's prompt and object-mask decoder is not used for this task.

Use the stored three-band IRRG imagery for SAM and omit DSM/nDSM. Apply the official SAM per-channel mean/std to the IRRG values while preserving their stored channel order. Record that IRRG is not natural RGB and that this input differs from PSPNet's five-channel /255 input.
