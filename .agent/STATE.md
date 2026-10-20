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
- The project folder in Google Drive currently contains the 106 PSPNet input bundle chunks, `parts_manifest.json`, and its README, with no training output, checkpoint, or evaluation metrics.
- The live Kaggle SAM kernel is `damminhtien/sam-vit-b-lora-six-class-potsdam-segmentation` and has status `KernelWorkerStatus.ERROR`. Its uploaded script hash matches the local `train_sam_kaggle.py`, but the attached Kaggle data bundle contains no `semantic_labeling` package. The first imports fail with `ModuleNotFoundError: No module named 'semantic_labeling'`, and Kaggle reports no output files. The local kernel metadata also contains a stale ID that differs from the live slug; this hindered status lookup but did not cause the runtime import failure.
- No cloud training completion, checkpoint, or held-out metrics were recovered. Do not include these attempts in the published evaluation results.
- Potsdam/Vaihingen inputs and prior PSPNet, SAM, and evaluation artifacts remain grouped under `/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/inputs/` and `runs/`. Legacy paths under `/Users/macbook/Data/input/` and `/Users/macbook/Data/runs/` remain symlinks for compatibility.
- Next step, if training resumes: fix the Kaggle package delivery and use a Colab run that persists checkpoints outside the ephemeral runtime, then verify saved outputs before reporting metrics.

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
