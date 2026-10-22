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
- Run `pspnet-colab-20261020-rerun` started at 2026-10-20 05:19 UTC on T4. Runner PID 36987 launched trainer PID 36988 from the verified epoch-2 checkpoint. The trainer completed epoch 3 and reached epoch 4 batch 400/1250, then exited with code -9 at 05:48:46 UTC. Colab no longer lists the session; no final metrics were written.
- The last checkpoint is saved locally at `/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/runs/pspnet-colab-20261020-rerun/last.pt` and in the Drive run folder. It is 787,996,002 bytes with SHA-256 `b1a4db629565971d76afcc66fc986735191c1139c417ab2271097bf50b3ff3d1`; the run manifest records `last_completed_epoch: 3`. Metrics are missing.
- LaunchAgent `com.codex.pspnet-colab-results-sync-20261020` synchronized the terminal error and epoch-3 checkpoint with size/SHA-256 verification, then exited on terminal status. The failed session was gone and CLI usage showed 0 compute units/0 active assignments before the user-authorized resume session was created.
- A new T4 session `pspnet-colab-20261020-resume-e3` is IDLE with no trainer process; `/content/drive` is not mounted. An explicit OAuth2 `colab drivemount` reached a Google Drive for desktop page that returned `Error 400: OAuth 2 parameters can only have a single value: state`. Direct `drive.mount()` through `colab exec` timed out because that path does not install the CLI Drive authorization callback. The connected Google Drive integration itself is authenticated and can access the project folder. The epoch-3 checkpoint remains durable locally and in Drive. Resume only after the new VM can access the exact input bundle and has a verified durable output-transfer path; preserve the failed run.
- On 2026-10-20, the user canceled further Colab use. The CLI transfer stopped at 58/169 parts before any trainer started; the fresh session was no longer found by the CLI. At the user's request, the 106 input bundle parts and `parts_manifest.json`/README were deleted from the project Drive root, along with the empty output folder created for this attempt. The prior failed run's Drive outputs, original local input bundle, and epoch-3 checkpoint remain preserved. Do not restart Colab unless explicitly requested.
- Kaggle kernel `damminhtien/sam-vit-b-lora-six-class-potsdam-segmentation` completed run `sam-vitb-lora-potsdam-20261020-rerun-02` at 2026-10-20 04:47 UTC. The run manifest reports 5/5 epochs, a successful CUDA smoke test on Tesla T4, verified dataset bundle SHA-256 `1f93e42f5464c9bf74c30f32ddefe8e175d7ccb679a3d84653f99aeb6eda29f1`, and runtime versions for Python, NumPy, PyTorch, torchvision, tifffile, Pillow, OpenCV, matplotlib, and tqdm.
- Kaggle metrics: validation OA 86.5103%, mIoU 62.7028%; held-out OA 82.7184%, mIoU 60.0213% over 14 scenes and 467,862,855 scored pixels. Held-out clutter/background IoU is only 0.0065%, so the aggregate score hides a severe failure on that class.
- Kaggle final checkpoint SHA-256 `01cf5018763633813c535beec8a4d469e36dd225bfbfeaa54e5576ee279fa64d` matches the run manifest. Metrics, run manifest, training log, checkpoint, overlays, execution log, and SHA-256 index are stored locally under `/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/runs/sam-vitb-lora-kaggle-output-20261020/snapshot-20261020-02/`. The 12 Kaggle artifact copies previously uploaded to Drive were deleted and the Drive run folder was verified empty.
- Historical failed attempts remain preserved; this run uses a distinct ID/path and does not overwrite them.
- The Colab portion of this task is canceled by the user; no resume run or new metrics exist. Kaggle's completed output remains local-only. The separate SAM accuracy sweep below is still in progress.

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

## SAM accuracy retrain sweep — complete

- Kaggle kernel `damminhtien/sam-vit-b-lora-six-class-potsdam-segmentation`, version 5, completed on Tesla T4. The executed trainer SHA-256 is `f387f8e05a652cfbde2bd5c20928669b0d317d75c66c29fd76015a4a56a716c5`; source package manifest SHA-256 is `179df412f05f536a0703b09da9f76f19870cab3be7148650b31c4b100025ca3c`; Potsdam bundle SHA-256 is `1f93e42f5464c9bf74c30f32ddefe8e175d7ccb679a3d84653f99aeb6eda29f1`.
- All three CUDA smoke tests passed. Validation mIoU: r4/512 = 73.1952%, r4/384 = 74.3216%, r8/512 = 73.3315%. Candidate B (r4/alpha8, patch 384, 455 steps/epoch) was selected by validation, best epoch 30. It improves the run-03 baseline 74.0987% by 0.2229 points.
- Candidate B's 14-tile heldout evaluation ran once after selection: OA 87.8142%, mIoU 73.7024% (run-03 baseline 73.1563%, +0.5461 points). Per-class IoU: impervious surfaces 83.2068%, buildings 86.2614%, low vegetation 74.2724%, trees 76.3332%, cars 85.1687%, clutter/background 36.9718%. Per-tile metrics and the confusion matrix are preserved in `candidate-b-r4-p384/metrics.json` and the local audit.
- Local output: `/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/runs/sam-vitb-lora-kaggle-output-20261020/snapshot-20261020-retrain-04/`. The sweep status is complete, heldout count is 1, the root artifact index verifies 34 files, the candidate indexes and best/last hashes verify, and the selected checkpoint's saved validation confusion matrix matches the validation metrics. The local `sam-vitb-lora-potsdam-20261020-retrain-04-local-audit.json` reconstructs candidate records and records the independent checks.
- The Kaggle root sweep manifest's `candidates` array is empty due to the executed script version. Candidate configs/scores are present in the three child run manifests and metrics; the original Kaggle artifacts remain unchanged. Commit `41eab3e` fixes recording candidate results for future sweeps; implementation and initial tests are in `b501abf`.

- The PSPNet Colab rerun was canceled by the user. No Colab session is active; do not resume without an explicit request.

## Data-pipeline migration — implementation status 2026-10-21

- The shared `deepnet.data` layer now provides immutable contracts, ID-based dataset catalogs, split manifests, geospatial validation, lazy raster windows, train-only statistics, sampling, aligned augmentation, scene evaluation, portable bundles, and run lineage. Generic named input specs keep model adapters configurable.
- Local Mac and Kaggle use the same source code and bundle format. Runtime selection prefers CUDA, then MPS, then CPU; CUDA-only pinned-memory and non-blocking transfer settings are enabled only for CUDA. The model adapter accepts a caller-created model and does not hardcode an architecture.
- Potsdam and Vaihingen source catalogs were built with SHA-256 outside the repository under `/tmp`; 38 Potsdam and 33 Vaihingen scenes were indexed. Potsdam IRRG validation passed with the frozen 18/6/14 split. Vaihingen IRRG validation passed. The Potsdam 5-channel view remains blocked by missing verified nDSM co-registration; Vaihingen's participant labels are absent for 17 scenes and require a future explicit split/label policy.
- The pure-Python wheel was built and installed with `--no-deps` in an isolated environment, preserving the existing Torch install. Its CLI and packaged dataset descriptors load, and `pip check` reports no broken requirements in that environment.
- Local Python 3.14.8 with Torch 2.14.1 selected MPS; a generic six-logit convolution forward pass through `TorchSegmentationAdapter` completed on MPS. The CUDA selection path and transfer settings are covered by mocked runtime tests; no Kaggle session or GPU training was launched.
- Focused validation passed: 20 synthetic GeoTIFF tests, Ruff checks and formatting, and Python compilation. Historical experiment metrics and notebook outputs are marked for rerun; the prior PSPNet/SAM training loops remain historical and do not yet consume the new bundles.
- `graphify update .` completed in code-only mode and rebuilt the ignored local index (752 nodes, 1,725 edges).
- Raw source rasters were only read during catalog creation. Catalogs and wheel artifacts remained under `/tmp`; no raw data or generated bundles were added to Git.

## Multimodal architecture follow-up — implementation status 2026-10-21

- Added versioned domain/band and augmentation contracts. Albumentations policies are injected into training datasets and operate on raw aligned bands, labels, and validity masks before model-view normalization; optical radiometry excludes auxiliary bands.
- Added configurable spectral input adaptation, a separate optical detail/elevation branch, an injectable dense-feature backbone protocol, a lazy Hugging Face wrapper, and a DPT-style four-scale decoder with a stride-four skip. RGB identity mapping and non-RGB learned projection are decided by config channel semantics.
- Added DINOv3 ViT-B/16 and DINOv2 ViT-B/14 model specs, an injected SAM feature-backbone baseline spec, Potsdam augmentation/domain configs, and a Potsdam run plan. Model weight loading is explicit; no pretrained weights were downloaded and no training was started.
- The five-channel Potsdam run plan remains gated on verified nDSM decoding, registration, and train-only elevation statistics. The IRRG-only model-input spec is available for a data view that does not use elevation.
- Synthetic unit tests passed for Albumentations API/config construction, synchronized geometry, deterministic optical-only radiometry, transformer prefix-token removal, 3/5-channel forward/backward, ignore-index loss handling, and patch-size 14/16 padding. A local MPS synthetic forward/backward smoke test also passed.
- A wheel built and installed with `--no-deps` in a temporary environment; installed config loaders successfully found packaged domain, augmentation, model-input, and model specs. No Kaggle GPU job was run.
- `graphify update .` rebuilt the ignored code index (933 nodes, 2,144 edges). The generated graph remains outside Git.

## DINOv3 split lock and full-scene data audit — 2026-10-22

- Locked the joint-run protocol in `configs/evaluations/isprs-dinov3-v1.json`. It binds the Potsdam and Vaihingen split hashes and training plan SHA-256 `b5e170e4f1192b587a6d97f8db798c5fcbc0bc6d07edadd4c4a006f394df7124`. Selection is the unweighted mean of the two dataset validation mIoUs, with separate dataset confusion matrices. Potsdam has no test split; Vaihingen scenes 2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29, 31, 33, 35, and 38 are reserved for one post-selection evaluation.
- The locked joint run uses three IRRG bands. It does not use the separate Potsdam five-band plan. The five-band source audit found 37/38 DSM metadata-grid matches and a one-column-short DSM at scene 3_13. All nDSM JPEGs lack CRS/affine and catalog co-registration evidence; 37 only match dimensions and 3_13 is also one column short. DSM/nDSM are outside the DVC input snapshot (0/76 files); nDSM decoding and DSM units/vertical datum need source confirmation.
- Full-scene IRRG histograms and per-scene statistics were generated for train/validation only: 38 Potsdam scenes and 16 Vaihingen scenes; every scanned IRRG pixel passed mask/finite checks. The audit did not read any Vaihingen test raster or label.
- The 19-cell notebook ran all 10 code cells after restarting the project `.venv` kernel. Its Jupytext source also completed in a fresh `.venv` process. Final reproducible artifacts and `eda-provenance.json` are under `/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/eda/20261022T061700Z-fresh-process/`; the interactive notebook run is under `/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/eda/20261022T061502Z/`.
- DVC status was up to date at lock SHA `17994ddb59a804bc847d3429cdd17dce1b0435db1c054e207e15565f28217b9c`. No DINOv3 training or Kaggle session was started.
