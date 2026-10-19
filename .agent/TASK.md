# Goal

Record the verified segmentation research and evaluation results in this repository, then push the completed documentation to the public GitHub repository. The requested push includes the six commits already on local `master` and the new report commit.

# Scope

- Publish aggregate frozen-encoder probe results for Potsdam, Vaihingen, LandCover.ai, and LoveDA.
- Record protocol, provenance, limits, leaderboard context, the provisional flagship recommendation, SAM status, and the sidewalk-segmentation research.
- Keep source imagery, labels, dataset archives, checkpoints, dense feature caches, predictions, previews, and runtime logs outside Git.
- Do not claim SAM or cloud training completion without verified outputs.

# Acceptance

- Reports and sanitized aggregate metrics are linked from the README.
- Metrics JSON files parse with Python 3.14 and contain no raw imagery, labels, weights, sample predictions, logs, or local filesystem paths.
- `git diff --check` passes and the staged diff contains only the intended report, index, invariant, and task-state files.
- Commit the documentation, push local `master` including all six existing commits, and verify GitHub `master` points at the resulting HEAD.
