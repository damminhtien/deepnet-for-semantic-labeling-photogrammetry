# Cloud rerun plan

## M1: Repair artifact and dependency delivery — complete

- Add a Colab runner that verifies the Drive chunks, prepares a unique local runtime bundle, and mirrors logs/manifests/checkpoints to Drive atomically throughout training.
- Package the two `semantic_labeling` modules used by SAM in a separate private Kaggle dataset, resolve and verify that package at runtime, and add a Kaggle dependency/GPU preflight.
- Correct Kaggle kernel metadata to the live kernel slug and retain the existing 18/6/14 Potsdam split.

Validation: hash-check staged source/data inputs; inspect the generated Kaggle source payload; run local syntax and import-boundary checks; `git diff --check`.

## M2: Resume PSPNet on Colab — cancelled by user

- The first run `pspnet-colab-20261020-rerun` was killed with exit code -9 during epoch 4 after completing epoch 3. The Colab session is gone; no final metrics were produced.
- The epoch-3 checkpoint is saved locally and in Drive (787,996,002 bytes; SHA-256 `b1a4db629565971d76afcc66fc986735191c1139c417ab2271097bf50b3ff3d1`). The old run and its artifacts remain preserved.
- A new T4 session `pspnet-colab-20261020-resume-e3` is IDLE. The CLI Drive mount's Google Drive for desktop authorization page returns `Error 400: OAuth 2 parameters can only have a single value: state`; the logged-in Google Drive connector works, but the Colab VM has no Drive mount and no trainer process.
- A direct `drive.mount()` through `colab exec` also timed out because it bypasses the CLI's Drive authorization callback. The user then canceled further Colab use. The CLI transfer stopped at 58/169 parts before any trainer started, and the session disappeared from the server.
- On the user's request, removed the 106 input bundle parts and their manifest/README from the project Drive root, plus the empty output folder created for this attempt. The prior failed run's Drive outputs, original local bundle, and epoch-3 checkpoint remain preserved. Do not resume Colab unless the user explicitly requests it.
- The prior local watcher polled every 120 seconds and saved changed Drive files with size/SHA-256 verification; it stopped after syncing the terminal error from the old run.

Validation: the old attempt's last durable epoch is 3/16. The fresh session disappeared before training; no trainer ran. The user-requested Drive cleanup left no input bundle parts or metadata in the project root. The prior failed run outputs and local source artifacts remain intact.

## M3: Launch SAM on Kaggle — complete

- Create the private source dataset, attach it alongside the existing Potsdam dataset, then push a new version of the existing live kernel.
- Confirm Kaggle reports the exact data and source datasets and observe the preflight before training.

Validation: kernel status/logs show dependency versions, CUDA GPU, verified bundle hash/split, and first epoch progress.

## M4: Monitor and verify results — Colab cancelled; Kaggle complete

- Kaggle completed 5/5 epochs and validation/held-out scoring. Outputs, execution log, checkpoint, previews, and SHA-256 index are stored locally only; the 12 Kaggle artifact copies previously uploaded to Drive were deleted and the Kaggle run folder was verified empty.
- Colab's first attempt ended after epoch 3/16 with trainer exit code -9. The resume was canceled before a trainer started; no new Colab metrics exist.
- Kaggle results remain verified locally; no Kaggle artifacts are stored in Drive.

Validation: final status and artifact hashes on Drive and Kaggle/local storage; no training artifacts staged in Git.

## M5: Improved SAM ViT-B + LoRA Kaggle rerun — complete

- Add train-frequency inverse-square-root class weights, weighted CE + macro soft Dice, separate LoRA/decoder AdamW rates, warmup/cosine decay, validation mIoU checkpoint selection, early stopping, and resumable `last.pt`/`best.pt` checkpoints.
- Keep the fixed 18/6/14 split, 512 patches, batch 1, seed 42, and 256 random patches per epoch. Evaluate all six validation tiles after each epoch and evaluate the 14 held-out tiles once after selecting `best.pt`.
- Verify the private source package and Potsdam bundle hashes before the run. Save only under `/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/runs/sam-vitb-lora-kaggle-output-20261020/snapshot-20261020-03/`; keep all training artifacts out of Git and Drive.
- Do not modify or restart the PSPNet Colab job in M2.

Validation: eight local unit tests pass; Ruff, Python compilation, CLI help, data bundle checksum, and remote source download/hash checks pass. Kaggle kernel version 4 completed all 30 epochs on Tesla T4; CUDA smoke passed. `best.pt` (epoch 26) was selected by validation mIoU, followed by one held-out evaluation. The complete manifest, metrics, both checkpoints, logs, previews, and SHA-256 index are local; all indexed artifact sizes and hashes were verified. Validation mIoU is 74.0987% and held-out mIoU is 73.1563%; artifacts remain outside Git and Drive.

## M6: SAM ViT-B + LoRA accuracy retrain sweep — in progress

- Keep run 03 as the historical baseline (validation mIoU 74.0987%, held-out mIoU 73.1563%). Train three fresh candidates on the fixed 18/6/14 split with seed 42, the existing weighted CE + macro Dice loss, AdamW schedule, 30-epoch limit, and patience 6.
- Mix uniform crops with 50% clutter-targeted crops and apply aligned right-angle rotations plus horizontal flips. Compare r4/512 (256 steps), r4/384 (455 steps), and r8/512 (256 steps, alpha 16). Keep validation and final inference tiling at 512.
- Complete smoke checks for all candidates, train each using validation only, choose by six-class validation mIoU (within 0.1 points prefer r4/512), then evaluate the selected checkpoint on the 14 held-out tiles exactly once.
- Use new run ID `sam-vitb-lora-potsdam-20261020-retrain-04`; store Kaggle output locally under `Data/.../runs/sam-vitb-lora-kaggle-output-20261020/snapshot-20261020-retrain-04/`. Leave the PSPNet Colab job and its OAuth-blocked resume state unchanged.

Checkpoint: Kaggle kernel version 5 was pushed with trainer SHA-256 `f387f8e05a652cfbde2bd5c20928669b0d317d75c66c29fd76015a4a56a716c5`; Kaggle reports `QUEUED`. The attached source package manifest hash is `179df412f05f536a0703b09da9f76f19870cab3be7148650b31c4b100025ca3c`, and the Potsdam bundle hash is `1f93e42f5464c9bf74c30f32ddefe8e175d7ccb679a3d84653f99aeb6eda29f1`. No GPU smoke result or training metrics are available yet.

Validation: three Kaggle CUDA smoke tests, validation metrics and checkpoint agreement, one held-out evaluation, and local artifact hash verification. Report an improvement only if the selected validation mIoU exceeds 74.0987%; keep all datasets, checkpoints, logs, and predictions outside Git.
