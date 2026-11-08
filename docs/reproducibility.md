# Reproducibility

Use the following levels to distinguish what a clean clone can verify from what still
requires external assets.

| Level | Meaning | Current status |
| --- | --- | --- |
| R1: metric reproduction | Recompute OA, class IoU/F1, and comparison tables from public confusion matrices. | Available in a clean clone. |
| R2: inference reproduction | Recreate predictions from the recorded checkpoint and evaluation data. | Final weights are public; source data and a configured runtime are still required. |
| R3: training reproduction | Recreate model weights from the source data, configs, and training code. | Code and configs are public; a clean-clone training rerun has not been verified. |

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

The four final state dictionaries used in the benchmark table are published in the
[model-weight release](model-weights.md), pinned to one Hub revision with SHA-256
checksums. The DINOv2, DINOv3, and PSPNet files contain full segmentation-model state;
the SAM file contains only its LoRA and decoder tensors, so inference also needs the
separate official SAM ViT-B base checkpoint. All files omit optimizer/scaler state,
source imagery, labels, and prediction rasters.

Inference still requires the correct ISPRS imagery, matching label/input conventions,
the repository code, and runtime dependencies. Training also requires the dataset
snapshot, matching label roles, pretrained initialization, and suitable compute. The
protocol and configs are documented under [`docs/experiments/`](experiments/README.md),
but archived author-side run outputs do not make training self-contained.

Some run exports identify their source by a snapshot SHA-256 rather than a Git commit;
others do not preserve a source snapshot digest. The evidence registry records missing
Git revisions as `null` and keeps any available snapshot digest separately. This limits
the ability to recreate the exact training checkout from public repository history.

The repository supports metric reproduction from public evidence (R1) and publishes the
final weights, but full clean-clone inference has not been verified. Dataset, upstream
model, and project-code terms are separate;
see [limitations](limitations.md#licensing-and-use) and the [weight release notes](model-weights.md#licenses-and-data-rights).
