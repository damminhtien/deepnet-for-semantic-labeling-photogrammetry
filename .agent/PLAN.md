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

## M6: SAM ViT-B + LoRA accuracy retrain sweep — complete

- Run 03 remains the reference: validation mIoU 74.0987%, held-out mIoU 73.1563%. All three fresh candidates used the fixed 18/6/14 split, seed 42, batch 1, 50/50 uniform/clutter-targeted crop sampling, aligned right-angle rotations/flips, the same weighted CE + macro Dice loss, AdamW schedule, and 512-pixel evaluation tiles.
- Validation results: A (r4/alpha8, train patch 512, 256 steps) 73.1952% mIoU, best epoch 29; B (r4/alpha8, train patch 384, 455 steps) 74.3216%, best epoch 30; C (r8/alpha16, train patch 512, 256 steps) 73.3315%, best epoch 16. B won by validation mIoU, improving run 03 by 0.2229 percentage points. Heldout was not used for selection.
- Candidate B was evaluated on the 14 held-out tiles once. Heldout OA is 87.8142% and mIoU is 73.7024%, up 0.5461 points from run 03. Per-class IoU: impervious surfaces 83.2068%, buildings 86.2614%, low vegetation 74.2724%, trees 76.3332%, cars 85.1687%, clutter/background 36.9718%.
- Kaggle kernel version 5 reports COMPLETE. All three CUDA smoke tests passed on Tesla T4. The Potsdam bundle SHA-256 is `1f93e42f5464c9bf74c30f32ddefe8e175d7ccb679a3d84653f99aeb6eda29f1`; the attached source manifest SHA-256 is `179df412f05f536a0703b09da9f76f19870cab3be7148650b31c4b100025ca3c`; the executed trainer SHA-256 is `f387f8e05a652cfbde2bd5c20928669b0d317d75c66c29fd76015a4a56a716c5`.
- Outputs are stored locally under `/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/runs/sam-vitb-lora-kaggle-output-20261020/snapshot-20261020-retrain-04/`. The root 34-file SHA-256 index, each candidate's artifact index, all best/last checkpoint hashes, selected checkpoint validation confusion, and aggregate metrics were verified. The per-tile heldout results and audit are in the local output.
- The original Kaggle `sweep_manifest.json` has an empty `candidates` array even though all three child run manifests and metrics contain their configurations and validation results. The original output was preserved; the local audit reconstructs the comparison and records the verification. Commit `41eab3e` fixes candidate tracking for future sweeps. The earlier implementation and tests are in commit `b501abf`.
- The PSPNet Colab job and its Drive OAuth-blocked state were not changed.

Validation: Python compilation, Ruff, and 18 unittest cases passed. Kaggle status is COMPLETE; the 14-tile heldout evaluation ran once after validation selection. Run 03 was exceeded on validation mIoU by 0.2229 percentage points. Training artifacts remain outside Git.
