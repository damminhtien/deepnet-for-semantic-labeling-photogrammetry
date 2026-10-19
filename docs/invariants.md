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

## Model and notebook changes

- Model definitions and experiment configuration still live in notebooks as well as `model_script/`. Check matching copies before changing model behavior; keep the shared data/evaluation helpers in `semantic_labeling/`.
- Model constructors no longer run at import time. Their runtime import has not been exercised in this checkout because PyTorch and torchvision are absent.
- Select validation based on the changed boundary. This repository has no root-level test configuration or dependency manifest in the current checkout; record the exact command and environment used instead of assuming a standard test command.
