# Data and experiment invariants

## Input and labels

- The model input has five channels in this order: IR-R-G, DSM, and normalized DSM. Most experiment notebooks use 256 by 256 patches; the SegNet notebook uses 224 by 224.
- `semantic_labeling/constants.py` defines the six output classes and the RGB palette. Class 0 is impervious surfaces (often described as roads); black maps to `IGNORE_LABEL = 6`, a no-boundary/undefined sentinel, not a seventh model class.
- Label decoding rejects unknown colors and IDs. Training loss and evaluation metrics ignore label 6, and metrics reject other out-of-range values. A batch containing only ignored targets produces a differentiable zero loss rather than NaN.
- Training and inference use `assemble_input` and `normalize_input` from the shared package. The current uniform `/255` scaling preserves the historical five-channel behavior. The source dataset describes DSM values as 32-bit floats, so this convention is centralized for consistency but is not claimed to be a calibrated physical normalization.
- `ISPRS_dataset` uses the supplied path templates, verifies that TOP/label and DSM/nDSM ID lists have matching lengths, honors the augmentation and cache options, and uses inclusive valid crop bounds.
- Sliding-window coordinates have one implementation; each edge-aligned window is emitted once, and the window counter delegates to the same generator. A step larger than the window size is rejected because it leaves image regions uncovered.
- The dataset description says ground truth for some benchmark scenes remains with the organizers. Do not use withheld evaluation labels or change benchmark splits as a workaround.

## Experiment evidence

- The README reports historical accuracy figures, but this checkout does not contain a run manifest or checkpoint that establishes those figures as reproducible results. Treat them as historical claims until a specific run can be tied to its code, data split, preprocessing, configuration, checkpoint, and metric output.
- For new evaluation results, record the scene IDs and split, preprocessing, model configuration, random seed when applicable, execution environment, checkpoint, and metric output.
- Keep training, validation, and benchmark evaluation claims separate. Do not infer performance from a code change or a successful static check.
- The 2026-10-19 foundation-model reports in `docs/experiments/` are separate, coarse-grid linear-probe experiments with aggregate metrics and provenance. They do not reproduce the historical PSPNet accuracy and are not official ISPRS benchmark submissions.
- Keep source imagery, labels, dataset archives, checkpoints, sample predictions, and rendered previews outside the public repository. Publish aggregate scores and provenance only.

## Model and notebook changes

- Model definitions and experiment configuration still live in notebooks as well as `model_script/`. Check matching copies before changing model behavior; keep the shared data/evaluation helpers in `semantic_labeling/`.
- Model constructors no longer run at import time. Their runtime import has not been exercised in this checkout because PyTorch and torchvision are absent.
- Select validation based on the changed boundary. This repository has no root-level test configuration or dependency manifest in the current checkout; record the exact command and environment used instead of assuming a standard test command.

## SAM cloud experiment

- Colab is assigned only to the existing five-channel PSPNet continuation. Kaggle is assigned only to the SAM ViT-B experiment; neither service should duplicate the other's model job.
- SAM initializes from the official ViT-B checkpoint. Its original image encoder parameters remain frozen; rank-4 LoRA updates are applied to each attention qkv/proj layer, and a separate semantic decoder outputs classes 0–5.
- SAM training inputs are the stored three-band IRRG pixels in their original order. The branch omits DSM/nDSM and applies SAM's pretrained per-channel mean/std normalization in the 0–255 scale. This differs from PSPNet's five-channel /255 contract and from SAM's natural-RGB pretraining.
- Use the fixed 18 training, 6 validation, and 14 held-out scene IDs. Training labels may come only from participant labels; no-boundary labels for validation and held-out scenes are reserved for scoring. Ignore class ID 6 in both loss and metrics.
- Keep Kaggle uploads private and exclude the PSPNet epoch-2 checkpoint, credentials, tokens, auth files, and notebook outputs from the SAM dataset.
- Record the official SAM source revision and checkpoint hash, data bundle hash, split, input semantics, LoRA configuration, decoder, runtime, and metric outputs with each cloud run.
