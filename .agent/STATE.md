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

## Carry-forward: separate cloud training remains unresolved

- On 2026-10-19 the user resumed PSPNet training in Colab session `pspnet-colab-20261019-rerun` on a Tesla T4. The new run is `pspnet-colab-20261019-rerun`, resuming the saved epoch-2 MPS model and Adam state; it targets epochs 3–16. The CUDA smoke test passed.
- The latest live probe reached epoch 3, batch 600/1250, with the process active and T4 utilization at 100%. No training epoch or held-out test has completed yet. At the observed rate, the remaining training is estimated at about 4.5–5 hours, plus validation and held-out evaluation; this is an estimate, not a completion guarantee.
- A macOS LaunchAgent at `~/Library/LaunchAgents/com.codex.pspnet-colab-results-sync.plist` runs the Python 3.14 sync process. It copies logs and manifests periodically and verifies each downloaded checkpoint's SHA-256 before keeping it under `/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/runs/pspnet-colab-20261019-rerun-02/`. Colab outputs are still pending.
- Potsdam/Vaihingen inputs and prior PSPNet, SAM, and evaluation artifacts are physically grouped under `/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/inputs/` and `runs/`. Legacy paths under `/Users/macbook/Data/input/` and `/Users/macbook/Data/runs/` remain symlinks for compatibility.
- The Kaggle SAM job's completion and metrics remain unverified. Do not imply that the published reports include this resumed PSPNet run or trained SAM results.

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
