# Reproducibility

Use the following levels to distinguish what a clean clone can verify from what still
requires external assets.

| Level | Meaning | Current status |
| --- | --- | --- |
| R1: metric reproduction | Recompute OA, class IoU/F1, and comparison tables from public confusion matrices. | Available in a clean clone. |
| R2: inference reproduction | Recreate predictions from the recorded checkpoint and evaluation data. | Not available from this repository alone. |
| R3: training reproduction | Recreate model weights from the source data, configs, and training code. | Configs and code are present; source data, gated weights, and run outputs must be obtained separately. |

## Recompute public metrics (R1)

The verification uses only Python's standard library. It validates registered SHA-256
hashes, verifies that the 14 per-scene matrices sum to the aggregate, and recomputes
OA, six class IoUs, F1, and the DINOv3–PSPNet scene comparison.

```bash
python3 scripts/recompute_public_metrics.py --check
```

To regenerate the metric tables in [`results.md`](results.md):

```bash
python3 scripts/recompute_public_metrics.py --write
```

The expected evidence is listed in [`artifacts/public/evidence-manifest.json`](../artifacts/public/evidence-manifest.json).
The public data snapshots are metadata only. No raster scan or training run is needed
for R1.

To validate local documentation links, report statuses, evidence IDs, and the generated
metric section together, run:

```bash
python3 scripts/check_public_docs.py
```

## Recreate figures and analyses

The paired-case and SciencePlots regeneration commands are documented in
[`figures.md`](figures.md) and the individual research reports. Some commands need the
external Potsdam cache, archived prediction maps, or Kaggle output files. They should
not be described as clean-clone reproductions while those inputs remain unavailable.

## Inference and training (R2/R3)

A new inference run needs the correct ISPRS data and trained model checkpoint. The DINOv3
pretrained snapshot requires access through its model host; final model weights and
prediction maps are not distributed in this repository. A training rerun additionally
needs the dataset snapshots, matching label roles, model initialization, and the
experiment's compute environment. The protocol and configs are documented under
[`docs/experiments/`](experiments/README.md), but archived author-side run outputs do not
make the training bundle self-contained.

Some run exports identify their source by a snapshot SHA-256 rather than a Git commit;
others do not preserve a source snapshot digest. The evidence registry records missing
Git revisions as `null` and keeps any available snapshot digest separately. This limits
the ability to recreate the exact training checkout from public repository history.

The repository therefore supports metric reproduction from public evidence (R1), not
full inference or training reproduction from a clean clone. Dataset and pretrained model
terms are separate from repository code terms; see [limitations](limitations.md#licensing-and-use).
