# Plan

## M1: Prepare and pause the PSPNet scratch run — complete

- Preserve PSPNet parameter names and random initialization.
- Run on MPS through epoch 2, save model and Adam state, then keep local training paused.

## M2: Analyze and prepare Potsdam data on the Mac — complete

- Verify the 18/6/14 split, five-channel input statistics, label decoding, and synchronized flips/mirrors.
- Preserve PSPNet input scaling and record the no-boundary 4_12 palette correction.

## M3: Build and verify the PSPNet cloud bundle — complete

- Package 38 scenes, prepared labels, source code, preprocessing metadata, and the epoch-2 checkpoint.
- Verify every bundle input with SHA-256.

## M4: Implement and stage the Kaggle SAM experiment — complete

- Add SAM ViT-B semantic segmentation with rank-4 attention LoRA on qkv/proj and a six-class decoder.
- Keep the original SAM encoder weights frozen; use the official checkpoint and per-channel normalization.
- Build and upload a private Kaggle dataset with three-band IRRG and labels for the fixed scene split.
- Stage a private GPU kernel that trains SAM only, separate from Colab's PSPNet job.

## M5: Authenticate and check cloud quotas — complete

- Complete manual Kaggle and Colab authentication without inspecting or printing credentials.
- Confirm Kaggle GPU quota and allocate the Colab T4 runtime.

## M6: Run the independent cloud jobs — in progress

- Colab verified all 169 bundle checksums, removed 38 non-manifest macOS sidecars, passed CUDA forward/backward smoke on a T4, and resumed from epoch 2. Continue through epoch 16.
- Kaggle trains SAM LoRA for five epochs of 256 random 512x512 patches, with a CUDA smoke run first. The kernel is currently queued.
- Save per-epoch checkpoints, logs, and runtime manifests.

## M7: Share the reusable bundle, evaluate, retrieve, validate, and commit — pending

- Upload the checksummed PSPNet archive to a private Drive folder and add a concise mount/extract guide for reuse from multiple Colab notebooks.
- Evaluate SAM on six validation and fourteen held-out scenes; save overall accuracy, mean IoU, and visual previews.
- Retrieve both cloud outputs under /Users/macbook/Data and compare the measured results with input/initialization caveats.
- Update Graphify, run focused syntax/style/whitespace validation, and commit only task-owned paths.
