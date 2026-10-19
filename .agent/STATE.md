# Current state

## Task

Document verified segmentation evaluations and research, then push the resulting reports and all user-authorized local `master` commits to GitHub.

## Status

- The report drafts and aggregate metrics are present under `docs/experiments/` and `docs/research/`.
- README and repository documentation link to the results and record the data boundary.
- Next: validate JSON and whitespace, inspect/stage explicit paths, commit, push `master`, and verify the remote ref.
- Local `master` contains six commits ahead of `origin/master`; the user explicitly said to push all of them along with this documentation.

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

The earlier split cloud experiment remains incomplete and was not resumed in this publication task. The last verified Colab log reached epoch 3 batch 200/1250, without a completed epoch 3; the last live check found no active Colab session. The Kaggle SAM job was queued, with CUDA execution and metrics still pending. Preserve this status if that cloud experiment is resumed; do not imply that these reports include PSPNet continuation or trained SAM results.

## Validation

- Both aggregate JSON files parse with Python 3.14.
- Reported frozen-probe and deep-evaluation primary metrics match the saved run JSONs for all three encoders.
- The metrics JSON artifacts contain no absolute local paths or sample predictions; raw datasets and model artifacts are not staged.
- `git diff --cached --check` passes. Project tests were not run because this task changes documentation and aggregate JSON only.

## Checkpoint

- Working branch: `master`.
- Current HEAD before report commit: `e1a7aec` (`docs: clarify Colab CLI auth and logging`).
- Exact next action: commit the report set, push all local commits, verify `git ls-remote origin refs/heads/master` equals the resulting HEAD, then mark this state complete.
