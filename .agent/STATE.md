# Current state

## Task

Document verified segmentation evaluations and research, then push the resulting reports and all user-authorized local `master` commits to GitHub.

## Status

- Complete: reports and aggregate metrics are published under `docs/experiments/` and `docs/research/`; README and repository docs link to them.
- Published report commit `4da7c9e` and all six prior local commits to GitHub `master`.
- Verified remote `master` at report commit `4da7c9eece479e903e2ff786b8bdfcfd4fe7ee51`; all six earlier commits are included in its history. The working tree was clean after that push.

## Verified experiments recorded

- Potsdam frozen-encoder 5 m validation probe and 10 m sensitivity check for UniverSat, AnySat, and TerraMind.
- Sampled RGB frozen-encoder probes on LandCover.ai v1 and LoveDA v1.
- Deeper Potsdam held-out and Vaihingen four-fold OOF plus official held-out evaluation for the same three encoders, with full/eroded references and cross-domain results.
- The reports distinguish the coarse scoring grids from full-resolution benchmark metrics. Source data, checkpoints, probe weights, feature caches, predictions, previews, and logs remain outside Git.

## Research status

- ISPRS Potsdam public results list the participant as `SWJ` with a 89.5% Overall score; this review did not find an official `SWJ_2` label. Other listed submissions exceed that score. The table's class-specific measures are F1 on eroded boundaries.
- TerraMind is documented as the provisional next prototype candidate, with UniverSat retained as challenger. No production-sized per-tile latency or peak-memory result is available.
- Tile2Net is the strongest direct sidewalk candidate found. Its 2023 paper metrics describe that paper's model and evaluation, not the current beta repository release.
- The SAM ViT-B LoRA implementation and private cloud run were prepared, but no completed training or held-out metrics were verified in the state available for this report.

## Carry-forward: cloud training status as of 2026-10-20

- The Colab session `pspnet-colab-20261019-rerun` is no longer present on the server (`No active sessions found`; session not found). Its manifest is stale and still says `running`.
- Exported Colab history shows the first training launch failed with `FileExistsError` because its run directory already existed without a manifest. A second launch succeeded at 2026-10-19 13:53 UTC and was still training at epoch 3, batch 800/1250, at the last observed progress. `last_completed_epoch` was null.
- The PSPNet script writes `last.pt` only after a full epoch. Run files were stored under ephemeral `/content`, not on Drive. The Mac watcher copied logs/checkpoints locally, but its last successful sync was 2026-10-19 14:08:09 UTC (22:08 Singapore time). The Mac booted again at 2026-10-20 12:00 Singapore time and launchd restarted the watcher at 12:01; there are no sync records across that gap. This explains why later run files were not copied to the Mac.
- The local Colab run directory has no checkpoint or metrics. The Colab CLI later pruned its stale local session record, but the retained history does not identify the remote runtime's termination reason. It is therefore unproven whether training stopped before completion or completed in the ephemeral runtime after the last sync.
- At the start of the current rerun, the project Drive folder contained the 106 PSPNet input bundle chunks, `parts_manifest.json`, and its README, with no output from the failed run.
- The live Kaggle SAM kernel is `damminhtien/sam-vit-b-lora-six-class-potsdam-segmentation` and has status `KernelWorkerStatus.ERROR`. Its uploaded script hash matches the local `train_sam_kaggle.py`, but the attached Kaggle data bundle contains no `semantic_labeling` package. The first imports fail with `ModuleNotFoundError: No module named 'semantic_labeling'`, and Kaggle reports no output files. The local kernel metadata also contains a stale ID that differs from the live slug; this hindered status lookup but did not cause the runtime import failure.
- No cloud training completion, checkpoint, or held-out metrics were recovered. Do not include these attempts in the published evaluation results.
- Potsdam/Vaihingen inputs and prior PSPNet, SAM, and evaluation artifacts remain grouped under `/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/inputs/` and `runs/`. Legacy paths under `/Users/macbook/Data/input/` and `/Users/macbook/Data/runs/` remain symlinks for compatibility.
- Next step, if training resumes: fix the Kaggle package delivery and use a Colab run that persists checkpoints outside the ephemeral runtime, then verify saved outputs before reporting metrics.

## Active task: cloud rerun requested 2026-10-20

- User requested a PSPNet Colab rerun with results saved to Drive and local project Data, plus a SAM Kaggle run whose results are saved locally only.
- Colab session `pspnet-colab-20261020-rerun` is mounted at `/content/drive`, read/write/rename/delete checks passed, and the accelerator is a Tesla T4. The Drive output folder is `MyDrive/deepnet-for-semantic-labeling-photogrammetry-data/runs/pspnet-colab-20261020-rerun`.
- All 106 input parts were SHA-256 verified and reconstructed to the expected 10,524,434,944-byte archive (SHA-256 `b0bc76f02ad827795e1eaa1fa50f4d75f1069461cc68c4336a60c307de60f88f`). The extracted bundle manifest and PSPNet trainer are present; the source checkpoint is epoch 2, 787,995,938 bytes, SHA-256 `ac30c7b806585bbd285c0faef03d0f9a992329fef045ae8a15e4e2c97c4bf427`; split is 18/6/14.
- Startup first stopped before training because the Drive preflight manifest had valid status `drive_mount_verified`, but the starter required `preflight_verified`. The starter now accepts both verified statuses, safely reuses the extracted bundle, verifies the checkpoint/trainer, and removes the reconstructed archive after setup.
- Run `pspnet-colab-20261020-rerun` started at 2026-10-20 05:19 UTC on T4. Startup returned runner PID 36987; the Drive runner manifest reports trainer PID 36988. At the last check the trainer process was alive, status was `running`, and its log reported `cuda:0`, Tesla T4, resume epoch 2, target epoch 16; no new epoch had completed yet.
- A new LaunchAgent `com.codex.pspnet-colab-results-sync-20261020` polls the Drive run folder every 120 seconds and downloads changed files with size/SHA-256 checks to `/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/runs/pspnet-colab-20261020-rerun/`. Initial setup/runner/preflight manifests and logs synced successfully. The stale 2026-10-19 watcher was unloaded; its files and data remain intact.
- Kaggle kernel `damminhtien/sam-vit-b-lora-six-class-potsdam-segmentation` completed run `sam-vitb-lora-potsdam-20261020-rerun-02` at 2026-10-20 04:47 UTC. The run manifest reports 5/5 epochs, a successful CUDA smoke test on Tesla T4, verified dataset bundle SHA-256 `1f93e42f5464c9bf74c30f32ddefe8e175d7ccb679a3d84653f99aeb6eda29f1`, and runtime versions for Python, NumPy, PyTorch, torchvision, tifffile, Pillow, OpenCV, matplotlib, and tqdm.
- Kaggle metrics: validation OA 86.5103%, mIoU 62.7028%; held-out OA 82.7184%, mIoU 60.0213% over 14 scenes and 467,862,855 scored pixels. Held-out clutter/background IoU is only 0.0065%, so the aggregate score hides a severe failure on that class.
- Kaggle final checkpoint SHA-256 `01cf5018763633813c535beec8a4d469e36dd225bfbfeaa54e5576ee279fa64d` matches the run manifest. Metrics, run manifest, training log, checkpoint, overlays, execution log, and SHA-256 index are stored locally under `/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/runs/sam-vitb-lora-kaggle-output-20261020/snapshot-20261020-02/`. The 12 Kaggle artifact copies previously uploaded to Drive were deleted and the Drive run folder was verified empty.
- Historical failed attempts remain preserved; this run uses a distinct ID/path and does not overwrite them.
- This task remains active until the Colab run completes or reaches a documented blocker, and its final outputs are verified on Drive and locally. Current milestone: runner and local/Drive persistence are active; Colab training has not completed its first epoch. Kaggle is complete and local-only.

## Validation

- Both aggregate JSON files parse with Python 3.14.
- Reported frozen-probe and deep-evaluation primary metrics match the saved run JSONs for all three encoders.
- The metrics JSON artifacts contain no absolute local paths or sample predictions; raw datasets and model artifacts are not staged.
- `git diff --cached --check` passes. Project tests were not run because this task changes documentation and aggregate JSON only.

## Checkpoint

- Working branch: `master`.
- Report publication commit parent: `e1a7aec` (`docs: clarify Colab CLI auth and logging`).
- Report commit `4da7c9eece479e903e2ff786b8bdfcfd4fe7ee51` was verified on `origin/master`; this completion-state commit follows it.
- Next task, if continued: resume the separate incomplete Colab PSPNet and Kaggle SAM runs; this report-publication task is complete.

## SAM rerun 03 — current status

- The improved trainer and tests are committed as `57a789c` and `c3ceeeb`. The training configuration uses patch size 512, batch 1, seed 42, 256 random patches per epoch, 30 maximum epochs, patience 6, AdamW, LoRA LR `1e-4`, decoder LR `3e-4`, weight decay `1e-4`, 5% warmup, and cosine decay to `1e-6`.
- The new loss is `0.7 * weighted CE + 0.3 * macro soft Dice`; training-only class frequencies determine inverse-square-root weights normalized to mean 1 and clipped to `[0.5, 2.0]`. Ignore label 6 is excluded from loss and metrics.
- Local validation passed: eight `unittest` cases, Ruff, Python compilation, CLI help, the full 77-file data bundle hash, and train-only frequency/weight computation. Bundle SHA-256 is `1f93e42f5464c9bf74c30f32ddefe8e175d7ccb679a3d84653f99aeb6eda29f1`.
- The source package SHA-256 is `dee63435e200731cc8ae435efd1bc0cabac7b6851ee2c5a1b161350cf3138e31`; a download from the private Kaggle dataset was unpacked and all four source files matched `source_manifest.json`. The initial upload attempt skipped the package directory; it was corrected with Kaggle directory ZIP mode and verified by downloading the resulting dataset.
- Kaggle kernel `damminhtien/sam-vit-b-lora-six-class-potsdam-segmentation`, version 4, completed run ID `sam-vitb-lora-potsdam-20261020-rerun-03` on a Tesla T4. The 512 x 512 CUDA smoke test passed with finite gradients.
- Training completed 30 epochs without triggering early stopping; the validation-selected checkpoint is epoch 26. Six validation tiles were evaluated after every epoch, and the 14 held-out tiles were evaluated once after checkpoint selection.
- Final validation: OA 90.0571%, mIoU 74.0987%. Final held-out: OA 87.8119%, mIoU 73.1563% over 467,862,855 pixels. Held-out class IoUs are 82.3111% impervious surfaces, 86.9149% buildings, 74.6866% low vegetation, 76.2911% trees, 86.5045% cars, and 32.2297% clutter/background.
- Relative to rerun-02, validation mIoU improved by 11.3960 percentage points and held-out mIoU by 13.1350 points. Held-out data was not used for model selection. Clutter/background remains the weakest class despite the large improvement from the baseline.
- The complete output is local under `/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/runs/sam-vitb-lora-kaggle-output-20261020/snapshot-20261020-03/`. `run_manifest.json` reports complete; best/last checkpoints match their manifest hashes, the selected checkpoint's saved validation confusion matrix matches final validation metrics, and all 12 entries in `artifact_sha256_manifest.json` passed size/SHA-256 checks. The downloaded Kaggle execution log has 14 held-out tile entries and no traceback.
- The existing PSPNet Colab state and job were not modified.
