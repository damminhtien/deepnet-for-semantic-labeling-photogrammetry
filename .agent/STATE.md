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
- The last successful local sync was 2026-10-19 14:08:09 UTC. The saved stdout reached epoch 3, batch 800/1250; `last_completed_epoch` was null. The local run directory contains only the manifest, stdout, sync state, and sync log, with no checkpoint or metrics. The next sync attempt on 2026-10-20 failed because the Colab status query exited with code 1.
- The project folder in Google Drive currently contains the 106 PSPNet input bundle chunks, `parts_manifest.json`, and its README. No training output, checkpoint, or evaluation metrics were found there. The last synced progress is the only recovered progress; whether the in-memory run advanced farther before the runtime disappeared is unknown.
- The live Kaggle SAM kernel is `damminhtien/sam-vit-b-lora-six-class-potsdam-segmentation` and has status `KernelWorkerStatus.ERROR`. Its log shows `ModuleNotFoundError: No module named 'semantic_labeling'` at startup, and Kaggle reports no output files. The local kernel metadata uses a different kernel ID and names only `train_sam_kaggle.py` as `code_file`; the staged local folder also contains the `semantic_labeling` package, indicating that the package was not available to the remote script.
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
