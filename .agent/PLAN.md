# Cloud rerun plan

## M1: Repair artifact and dependency delivery — complete

- Add a Colab runner that verifies the Drive chunks, prepares a unique local runtime bundle, and mirrors logs/manifests/checkpoints to Drive atomically throughout training.
- Package the two `semantic_labeling` modules used by SAM in a separate private Kaggle dataset, resolve and verify that package at runtime, and add a Kaggle dependency/GPU preflight.
- Correct Kaggle kernel metadata to the live kernel slug and retain the existing 18/6/14 Potsdam split.

Validation: hash-check staged source/data inputs; inspect the generated Kaggle source payload; run local syntax and import-boundary checks; `git diff --check`.

## M2: Launch PSPNet on Colab — running

- The named T4 session has a verified Drive mount and write/read access. Run `pspnet-colab-20261020-rerun` started from the verified epoch-2 checkpoint; its detached trainer process and runner manifests are live.
- Drive is the durable Colab output location. A new local LaunchAgent polls the run folder, verifies each downloaded file by size and SHA-256, and saves it under project Data. The stale watcher for the 2026-10-19 session is unloaded.
- The first startup attempt stopped safely because the starter accepted only `preflight_verified`, while the valid manifest said `drive_mount_verified`. The starter now accepts both verified states and reuses the already validated extracted bundle without assembling the 10.5 GB archive again.

Validation: runner PID 36987, trainer PID 36988, Tesla T4, verified source checkpoint epoch 2, Drive manifests and initial logs downloaded locally. Training has not yet recorded its first completed epoch; do not start a duplicate on CLI timeout.

## M3: Launch SAM on Kaggle — complete

- Create the private source dataset, attach it alongside the existing Potsdam dataset, then push a new version of the existing live kernel.
- Confirm Kaggle reports the exact data and source datasets and observe the preflight before training.

Validation: kernel status/logs show dependency versions, CUDA GPU, verified bundle hash/split, and first epoch progress.

## M4: Monitor and verify results — in progress

- Kaggle completed 5/5 epochs and validation/held-out scoring. Outputs, execution log, checkpoint, previews, and SHA-256 index are stored locally only; the 12 Kaggle artifact copies previously uploaded to Drive were deleted and the Kaggle run folder was verified empty.
- Colab training is running. Its final metrics and checkpoint remain pending; the local watcher and Drive mirror are active.
- Report completed metrics only when final manifests, checkpoints, and evaluation outputs agree.

Validation: final status and artifact hashes on Drive and Kaggle/local storage; no training artifacts staged in Git.
