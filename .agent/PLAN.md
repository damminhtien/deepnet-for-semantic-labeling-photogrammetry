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

- The original Colab session was pruned by the backend before epoch 3 completed; its last known log reached epoch 3 batch 200/1250. A live check on 2026-10-19 found no active Colab sessions, and `pspnet-colab-20261019-retry` was not found. The open tab is an empty `scratchpad` with no connected runtime. Reconnect a T4, mount the private Drive folder, verify/reconstruct the archive, then resume from epoch 2.
- Kaggle trains SAM LoRA for five epochs of 256 random 512x512 patches, with a CUDA smoke run first. The kernel is currently queued.
- Save per-epoch checkpoints, logs, and runtime manifests.

## M7: Share the reusable bundle, evaluate, retrieve, validate, and commit — in progress

- Completed: uploaded the PSPNet archive as 106 numbered 95 MiB parts, plus `parts_manifest.json` and a mount/reconstruct/extract README, to the private Drive folder. Drive readback found all expected names and sizes; the remote manifest matches the local manifest byte-for-byte.
- Evaluate SAM on six validation and fourteen held-out scenes; save overall accuracy, mean IoU, and visual previews.
- Retrieve both cloud outputs under /Users/macbook/Data and compare the measured results with input/initialization caveats.
- Update Graphify, run focused syntax/style/whitespace validation, and commit only task-owned paths.

## M8: Prevent Drive OAuth/ADC confusion — complete

- Document that Drive file work uses the connected Drive integration, while gcloud ADC is a separate credential with independent scopes.
- Record the handling for `403 ACCESS_TOKEN_SCOPE_INSUFFICIENT` and keep Colab's in-runtime Drive mount separate.
- Validation: inspect the workflow/decision/state documentation and run `git diff --check`.

## M9: Separate Colab CLI auth from browser and VM auth — complete

- Verify the existing ADC path against the Colab assignments endpoint, distinguish an empty session list from an authentication failure, and inspect the open notebook state.
- Configure the user-level Colab CLI command to default to ADC while preserving explicit `--auth` choices; redact authorization values from the local CLI log.
- Record the auth boundary and avoid verbose HTTP logging for authenticated requests.
- Validation: `colab sessions` and `colab --auth adc sessions` both return the authenticated empty result; inspect the notebook UI and verify the CLI log contains no unredacted authorization values.
