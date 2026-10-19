# Goal

Complete the PSPNet scratch-training experiment with two independent GPU runs on Colab and Kaggle. The Mac is for data analysis, preprocessing, augmentation checks, and postprocessing; local training is paused after epoch 2.

# Constraints

- Preserve the historical `model_kernel/pspnet_2.ipynb` and its saved outputs.
- Preserve the 18/6/14 train/validation/held-out scene split, six output classes, ignore label 6, and project `/255` input scaling.
- Continue both cloud runs from the same locally saved epoch-2 model and Adam checkpoint; identify them as independent replicas, not distributed training.
- Keep the Kaggle dataset private. Do not publish the ISPRS data or upload credentials, tokens, or auth files.
- Keep checkpoints, logs, manifests, and prepared data under `/Users/macbook/Data`, outside Git.
- Preserve unrelated workstation state and stage only task-owned files.
- No historical PSPNet checkpoint is available; report the outcome as a new scratch run rather than an exact reproduction of the historical score.

# Acceptance

- Local preprocessing and augmentation checks complete, with bundle files verified by SHA-256.
- Both cloud jobs pass a real CUDA forward/backward smoke run and finish epochs 3–16.
- Validation and held-out metrics, checkpoints, logs, and runtime provenance are downloaded under `/Users/macbook/Data`.
- Compare the two independent runs and report the measured results and any divergence.
- Update Graphify, run focused syntax/style/whitespace validation, and commit only task-owned paths.
