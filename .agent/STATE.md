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

## Rare-class augmentation audit — 2026-10-22

- Expanded the EDA notebook and Jupytext source with a train-only scan of 12,121 overlapping candidate windows across 18 Potsdam and 12 Vaihingen training scenes. Rare classes are defined as at most 6% of train pixels or present in at most half of train scenes; selected classes are car and clutter/background.
- Selected five crops per dataset: high-density and near-10th-percentile positive-density windows for each rare class, plus a boundary-rich window. Replayed each crop with eight deterministic seeds, for 40 augmentation trials per dataset. No validation or official test scenes were used for augmentation selection/replay.
- Potsdam car pixels disappeared in 2/16 trials on one low-density `6_7` crop containing 1,274 source car pixels. Other evaluated class trials had no complete disappearance. Class-edge fractions were measured from 4-neighbor semantic boundaries; same-class instance boundaries are unavailable in these labels.
- Affine padding was correctly marked invalid with ignore labels in all trials. Photometric transforms nevertheless changed padded optical values in 30% of Potsdam trials and 25% of Vaihingen trials. Mean nonzero fraction within padded pixels rose from 2.93% to 27.08% and from 8.23% to 33.33%, respectively. Model patch-stride padding was zero.
- Do not promote `isprs-irrg-aerial-v2` to the full run yet. Restore the configured image fill after radiometry (or make it mask-aware), review a rare-class sampling/bounded-geometry policy, and rerun this focused audit. The augmentation and training configs remain unchanged.
- External outputs and provenance are at `/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/eda/20261022T063946Z/`, including rare-boundary examples, audit summaries, and DVC/split/config hashes. The project `.venv` kernel was restarted; analysis cells and final export source ran through Jupyter MCP. Ruff and Python compilation passed. `graphify update .` refreshed the ignored code index (1,032 nodes, 2,393 edges). No training was run.


## Potsdam benchmark split and evaluation protocol — 2026-10-22

- Official ISPRS Potsdam uses 24 reference-labeled tiles for training or internal evaluation and reserves 14 tiles for organizer test evaluation. ISPRS does not define an internal validation assignment. DINOv3 v2 uses the fixed project 18/6 split within the 24 labeled tiles and preserves the 14 challenge-heldout IDs as test. DINOv3 v1 configs remain unchanged.
- Active configs: `configs/splits/potsdam-dinov3-v2.json`, `configs/evaluations/isprs-dinov3-v2.json`, and `configs/training/isprs-dinov3-joint-v2.json`. The v2 protocol binds training/split hashes and requires checkpoint selection on validation only. Potsdam table-style scoring is per-class F1 for IDs 0–4 plus overall accuracy from no-boundary references; save per-tile and summed confusion matrices, with all-six-class metrics as supplements. Local references are scene-matched and hashed, but equivalence to organizer-held scoring files is unverified. The 14 tiles were used in separate historical experiments and v1 EDA; describe v2 as benchmark-aligned held-out evaluation, not a never-used program test.
- The default `dvc.yaml` exports 18/6 Potsdam and 12/4 Vaihingen train/validation bundles only. `pipelines/evaluation/dvc.yaml` defines isolated test-only bundle targets. Potsdam's 14 eroded references are present in its DVC source. Vaihingen's 17 test references remain outside the current DVC source snapshot and must be imported with its four validation labels before its evaluation target can run.
- `.dvcignore` excludes Python bytecode from the `deepnet/data` source hash. `uv run dvc repro` completed for the default train/validation graph; both quality reports passed, and bundle hashes remained stable. Split/config binding hashes and bundle role counts were checked. `uv run dvc status` reports the optional evaluation pipeline's test outputs as unbuilt, as intended; its test stages were not executed.
- The prior EDA statistics came from the v1 18/20 Potsdam split and are now marked historical; regenerate v2 EDA outputs before using validation distributions for tuning. The evaluation helper supports full-scene tiled inference and confusion summaries, but a complete model-loading training/evaluation caller is still required. No training or v2 test evaluation was run; the test metric test cases were edited but not executed.

## Held-out data readiness — 2026-10-22

- Completed Vaihingen test-label staging from the locally available ISPRS eroded references. The DVC import now contains the 4 validation and all 17 test labels (21 files, 329,638,420 bytes); each validation source label matched the existing DVC copy by SHA-256. Potsdam already had all 14 test eroded labels in its DVC source.
- Rebuilt the Vaihingen catalog and default train/validation bundle. Added test-only DVC quality stages for both datasets, then validated and exported the isolated test bundles. The default bundles still contain Potsdam 18/6 and Vaihingen 12/4 only.
- Test membership is exact: Potsdam 14/14 and Vaihingen 17/17. Test quality reports passed across 504,000,000 Potsdam pixels and 90,200,065 Vaihingen pixels; valid-pixel fraction is 1.0 in both. All four train/validation and test bundles passed `deepnet-data verify-bundle`; `uv run dvc status` reports up to date.
- DVC remote `storage` received 31 files. No model inference, test scoring, training, or Kaggle session was started. The training/evaluation caller and the previously identified augmentation padding correction remain prerequisites for the full run.
- The earlier benchmark note saying Vaihingen test references were absent is superseded by this section. Historical local DVC source staging is `/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/dvc/staging/vaihingen-eroded-evaluation-labels`.

## Seven audit repairs — 2026-10-28

- User authorized all seven fixes from the repository audit, with tests only where
  necessary. See TASK.md and PLAN.md for the bounded scope and verification steps.
- Initial HEAD: `6084012`. The existing unrelated PSPNet wording edit in this file
  is preserved and must not be included in these commits.
- Fix 1 complete: count-based F1; 48 published macro-F1 fields verified, 33
  corrected, source counts retained, correction reproducible and idempotent.
  Five focused tests and Ruff passed. All other existing report fields unchanged.
- Fix 2 complete: quality scans every required image and its metadata, intersects
  raster/finite masks per view, and separates valid from scored pixels. Regression
  cases cover missing/shifted images, nodata, NaN, and independent view masks.
  All 33 data-pipeline tests passed; Ruff passed.
- Fix 3 complete: evaluator returns each scene confusion matrix and the aggregate.
  The existing full-grid evaluation test now checks their equality; test and Ruff
  passed.
- Fix 4 complete: per-scene temporary directories are removed before the next
  scene and on inference failure. The existing evaluator test now uses two scenes,
  checks summed matrices, and verifies scratch cleanup on success and failure.
  Focused test, Ruff, and formatting passed.
- Fix 5 complete: both raster APIs share one read path. WarpedVRT reads are
  clipped to the reference grid, data and masks are padded explicitly, and alpha
  coverage excludes areas outside the source footprint even without source nodata.
  Both padded/in-bounds paths passed with and without nodata; all 35 pipeline tests
  and Ruff passed.
- Fix 6 complete: lineage uses datetime.timezone.utc, retaining Python >=3.10.
  Built and installed the package with only core dependencies in an isolated uv
  Python 3.10.22 runtime outside the project. Package import, UTC timestamp, and
  CLI help passed; two existing bundle/lineage tests and Ruff passed.
- Fix 7 complete: the active protocol and documentation describe Vaihingen as
  scene-held-out evaluation after joint training. The training configuration and
  split memberships are unchanged; their protocol hashes were verified.
- Final core checks: 58 tests and 2 subtests passed in tests/unit plus the existing
  SAM loss test. Ruff check and format check passed for deepnet, tests/unit,
  scripts, and notebooks_src. Graphify code update completed (1063 nodes).
- Rechecked all 48 published macro-F1 records directly from their saved matrices;
  33 corrections and no changes to any other pre-existing report field. Original
  metric-file SHA-256 hashes still match the correction provenance.
- Default DVC pipeline rerun passed. Quality scans covered 24 Potsdam and 16
  Vaihingen train/validation scenes, including required imagery; validity is 1.0.
  Both rebuilt training bundles passed all file hashes and embedded contracts;
  content fingerprints are unchanged and test scenes remain excluded.
- Isolated test DVC pipeline rerun passed for all 14 Potsdam and 17 Vaihingen
  scenes. Potsdam has 504,000,000 valid / 467,862,855 scored test pixels;
  Vaihingen has 90,200,065 valid / 81,991,943 scored test pixels.
- All four rebuilt bundles passed file SHA-256, inventory, embedded contracts,
  exact role membership, and comparison with the previous manifests in DVC cache.
  Every bundle content fingerprint is unchanged; source_revision records the
  repaired code at 161bc06. Both DVC pipelines report up to date.
- Validation boundary: the initial full-suite collection remains blocked in five
  historical SAM modules by missing torchvision. No full scientific model run was
  performed. Original external artifacts and the unrelated working-tree edit are
  preserved; no training, cloud jobs, or remote pushes were started.
- Complete: all seven repairs and final verification are committed locally.

## DINOv2 full run — 2026-10-31

- User explicitly requested the full training flow and DINOv2 training. Initial
  HEAD is 4ff585b; the pre-existing PSPNet wording edit remains unrelated.
- Kaggle CLI authentication works. Local runtime is Python 3.13 / Torch 2.14.1
  with MPS, Transformers 5.20.2, PEFT 0.21.2, and Albumentations 2.0.8.
- Existing DINOv2 attention module names match the configured LoRA suffixes in the
  installed Transformers. Public facebook/dinov2-base is ungated at revision
  f9e44c814b77203eaa57a6bdbbd535f21ede1415 (observed via Hugging Face API).
- Milestone 1: radiometry restores the exact pre-radiometry values at invalid
  pixels, preserving configured affine fill. New policy isprs-irrg-aerial-v3 uses
  bounded affine ranges (scale 0.95–1.05, rotation ±6°, translation ±2%, p=0.25).
  Eight augmentation tests passed, including zero/nonzero fill regressions.
- Replayed 80 train-only rare-class/boundary trials at the actual 504/280 windows:
  17 trials had invalid pixels, zero invalid-fill changes, and zero disappearance
  of a source-present car/clutter class. External report is
  /Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/runs/isprs-dinov2-joint-v1-20261031-preflight/augmentation-audit.json.
- Downloaded the pinned public pretrained snapshot to the external models folder;
  model.safetensors SHA-256 is d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841.
- Next: finish and validate the shared training/evaluation caller.
- Milestones 2–3: added the shared joint runner and pinned DINOv2 v2 model spec,
  training config, and locked evaluation protocol. It persists optimizer/scaler/RNG
  checkpoints, exact partial-epoch histograms, validation confusion matrices, and
  frozen checkpoint selection before any test-bundle access.
- Real pretrained MPS preflight passed on one augmented training crop per domain:
  loss 1.5064847469, 11,777,170 trainable parameters. Output is outside Git at
  /Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/runs/isprs-dinov2-joint-v1-20261031-preflight/local-mps-step.
- Focused core validation: 63 tests and 2 subtests passed, including exact
  uninterrupted/resumed weight equality and one-time post-selection scoring.
  Ruff check/format passed. Graphify code update produced 1,120 nodes/2,823 edges;
  generated output stays ignored. DVC default and isolated test pipelines reran;
  all quality reports passed, all four bundle content hashes stayed unchanged,
  and dvc status is up to date. No trained-model accuracy is claimed yet.
- Prepared and uploaded three new private Kaggle datasets (all version 1):
  damminhtien/isprs-irrg-train-validation-20261031,
  damminhtien/isprs-irrg-heldout-20261031, and
  damminhtien/dinov2-joint-source-20261031. All are ready; metadata confirms
  isPrivate=true. Remote expanded inventories and byte sizes match all payloads.
- Kaggle auto-extracts tar.gz inputs. Bootstrap now verifies all 167 expanded
  payload files against locally frozen SHA-256 inventories, then links the
  read-only inputs into the expected layout. Normal archive inputs remain supported.
  Local corruption/traversal rejection checks passed. No GPU pilot was launched.
- Uploaded runner wheel comes from 885535f, SHA-256
  b0b9cd21a24d7763cad02ebc93f9cb5e440a1b943e389a2340473c89d6a5bf8d.
  Bootstrap changes are separately committed; source inputs remain immutable.
- Added a local watcher to retrieve terminal outputs and verify artifact hashes,
  checkpoint selection, exactly-once test status, and summed per-scene counts.
  Its audit passed on the synthetic end-to-end training output; all 3 runner
  tests and relevant Ruff checks passed after adding this audit.
- Milestone 4: launched private Kaggle kernel version 1 on requested
  NvidiaTeslaT4, 12-hour platform timeout, full 20-epoch configuration:
  https://www.kaggle.com/code/damminhtien/dinov2-joint-irrg-segmentation-20261031.
  Remote pulled bootstrap SHA-256 matches the staged source; GPU/private flags and
  all three attached inputs are verified. Initial observed status is QUEUED.
- Persistent local watcher PID 21591 polls once per minute, captures remote logs,
  and downloads/audits terminal artifacts under the external run folder's monitor/.
  Exact source/config/bootstrap hashes and launch identity are in
  /Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/runs/isprs-dinov2-joint-v1-20261031-kaggle/cloud-run.json.
  Final model metrics do not exist yet; milestone 5 remains pending.
- Final local core check after adding the retrieval audit: 63 tests and 2 subtests
  passed in 5.15 seconds. The watcher is independent of the foreground agent;
  caffeinate PID 21761 prevents idle sleep while watcher PID 21591 exists.
  GPU allocation remains queued as of 2026-10-31 14:48:45 UTC; no cloud training
  loss or final model metrics have been observed yet.

## DINOv2 result review and runtime repair — 2026-11-01

- Live Kaggle check confirms attempt 1 / kernel version 1 ended ERROR. Watcher
  observed RUNNING at 2026-10-31 23:05 SGT and ERROR at 23:08 SGT, retrieved six
  files, and exited. No checkpoint, optimizer update, selection, or model metrics
  exists; the failure occurred during PEFT LoRA construction, before training.
- All three input groups passed bootstrap SHA-256 checks. Actual accelerator was
  Tesla T4 with Torch 2.11.0+cu128. The traceback is explicit: PEFT 0.21.2 found
  preinstalled torchao 0.10.0 but requires at least 0.16.0 for that optional backend.
- Independently verified both indexed failure artifacts and the downloaded source
  manifest. Preserved original files unchanged. Derived failure review is under
  the original external run folder at review-20261101/failure-review.json.
- Fixed bootstrap to remove unused torchao in the ephemeral Kaggle runtime, retain
  accelerator package versions, and perform an actual CPU LoRA forward/backward
  preflight. It records package versions/removal and a distinct attempt identity.
- Reproduced the exact PEFT error using a fake old backend in an isolated temporary
  environment; the real removal path then passed actual PEFT LoRA forward/backward.
  Four training/audit tests and Ruff passed. Failed-run audits now verify hashes
  and report failed explicitly rather than returning incomplete with empty errors.
- Continue the original authorized full-training task using the frozen config,
  protocol, source wheel and all three existing input datasets. A new kernel and
  local attempt folder will preserve attempt 1; scientific results remain pending.
- Attempt 2 launched as a separate private kernel, version 1:
  https://www.kaggle.com/code/damminhtien/dinov2-joint-irrg-segmentation-20261101.
  Remote bootstrap source hash/private/GPU/input metadata verified. Runner wheel
  remains frozen at 885535f; bootstrap repair is 5f94d35. Training/protocol/input
  hashes are unchanged; the experiment ID stays fixed and attempt/output IDs differ.
- New attempt folder:
  /Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/runs/isprs-dinov2-joint-v1-20261101-attempt-02.
  Watcher PID 2836, idle-sleep guard PID 2837, live-log stream PID 2869. All
  three input groups passed cloud hashes. Removed torchao 0.10.0 successfully;
  actual LoRA CPU forward/backward passed inside the T4 runtime.
- Full DINOv2 model initialization passed the previous failure point. Actual
  GPU optimizer update 1 completed at 2026-11-01 12:41:33 SGT, loss 1.512652.
  Epoch 1 reached 64/64 updates at 12:42:11 SGT, last update loss 1.390912;
  full-scene Potsdam validation began at 12:42:13. No held-out test scoring yet.
- Local final core validation after the repair: 64 tests and 2 subtests passed
  in 5.38 seconds; Ruff check/format and diff checks passed. Graphify code update
  completed with 1,132 nodes/2,870 edges, generated output remains ignored.
- Milestone 5 now records verified recovery/GPU progress. Milestone 6 remains
  pending for final artifacts and scientific assessment; training loss alone
  does not establish validation/test accuracy or convergence.

## DINOv2 validation comparison — 2026-11-01

- Attempt 2 completed all 20 epochs / 1,280 optimizer updates. Final validation
  completed at 16:20:17 SGT. Selected epoch 15: Potsdam mIoU 51.0339%, OA 77.9859%;
  Vaihingen mIoU 47.6903%, OA 79.1191%; selection mean 49.3621%. These are rounded
  live-log observations, not yet downloaded full-precision artifacts.
- Potsdam test began at 16:21:34 SGT. Fresh watcher status at 16:37 SGT remains
  RUNNING. Preserve the selected checkpoint and one-time test protocol. Milestone
  6 stays pending; per-class metrics and final hash audit are not available yet.
- Directly inspected the historical SAM candidate-b-r4-p384 metrics, manifest and
  all 30 validation records. Its Potsdam validation mIoU is 74.321623%, a 23.2877
  point gap. Scene IDs, per-scene scored counts and all six GT class supports match
  current quality records. Historical non-overlapping inference and current
  weighted-overlap inference still differ; this is not a controlled backbone test.
- Total trainable parameters are 618,182 for SAM and 11,777,170 for DINOv2. Full-run
  Potsdam input pixel presentations are 2.013 billion versus 0.650 billion. At
  roughly matched exposure SAM epoch 10 already achieved 68.8812% versus DINOv2
  epoch 20's 50.9364%; budget alone is an insufficient established explanation.
- Source review found raw intermediate DINOv2 hidden states without the reference
  final LayerNorm, randomly initialized IRRG projection, a larger unnormalized
  convolutional decoder, and different joint-domain sampling. These are concrete
  differences and candidate causes, not measured causal effects. No new training
  or test rerun was launched. Do not infer overfitting or class collapse yet.
- Review: docs/experiments/isprs-dinov2-joint-validation-review-2026-11-01.md.
  External analysis-20261101/ contains comparison JSON, all 20 parsed observations,
  reproducible Matplotlib PNG/PDF and SHA-256 evidence. Four data-comparison checks
  passed; the figure was visually inspected. Only analysis/docs changed.

## DINOv2 final retrieval and scientific assessment — 2026-11-01

- Fresh Kaggle status is COMPLETE. Scientific evaluation completed 16:47:46 SGT;
  the full flow exited successfully at 16:47:53. The old watcher had stopped before
  completion. Restarted retrieval only; all outputs downloaded/audited by 17:03:45.
  No new training, local inference, held-out rerun or configuration change.
- External attempt-02 monitor/outputs/ contains both checkpoints and the complete
  original output. Its index verifies 12 artifacts / 882,900,091 bytes; supplied
  and independent audits have errors: []. Recomputed 45 per-scene/aggregate metric
  matrices, verified source/configuration/protocol/binding hashes, checkpoint
  progress and selection, and one completed test evaluation per dataset.
- Selected best.pt is epoch 15 / step 960, SHA-256
  22c5c4bd342b79cc767a2f8c8f4c03c73e09c60ee14d5803250b5573f0b033f9.
  last.pt is epoch 20 / step 1,280 with optimizer, AMP scaler and RNG state. Frozen
  selection is the maximum recorded validation mean; all 48 LoRA B tensors updated.
- Potsdam test: 48.652261% mIoU, 74.657628% OA, 467,862,855 pixels / 14 scenes.
  Vaihingen test: 49.178756% mIoU, 78.303265% OA, 81,991,943 pixels / 17 scenes.
  Validation remains 51.033863% Potsdam and 47.690299% Vaihingen at epoch 15.
- Potsdam test gap to historical SAM is -25.0501 mIoU points. Matching scene IDs,
  scored counts and GT class supports; lower scores on all 14 scenes. Different
  inference/training recipes still prevent an isolated backbone comparison.
- Test class IoUs (impervious/building/low vegetation/tree/car/clutter): Potsdam
  62.7431/58.0959/62.9809/65.9835/39.9906/2.1195%; Vaihingen
  67.2368/68.6921/55.6821/70.8283/21.3713/11.2618%. Potsdam clutter recall is
  2.2018%; only 1.212M predicted versus 19.920M truth pixels. Car and clutter
  account for 53.25% of the gap, but every class regresses against SAM.
- Training histograms show clutter was present in 1,781/2,560 Potsdam patches
  (44.342M pixel presentations, 6.88%). Missing all clutter training samples is
  ruled out. Checkpoint spectral projection's third row is entirely negative;
  its normalized unit-input bounds are [-7.9782, -1.7615]. This confirms an input
  distribution concern; causal accuracy effects and other interface hypotheses
  still need future validation-only controls.
- Final report: docs/experiments/isprs-dinov2-joint-final-review-2026-11-01.md.
  review-final-20261101/ holds independent-audit.json, CSVs, exposure/adapter data,
  reproducible Matplotlib figures and review-sha256.json. Both PNGs were visually
  inspected. Milestone 6 is complete; preserve artifacts/protocol and do not start
  further training without a new experiment decision/request.
