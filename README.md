# DeepNet: an auditable pipeline for aerial semantic segmentation

DeepNet is a research engineering project for preparing, training, and evaluating
single-model semantic segmentation systems on geospatial imagery. The repository brings
data integrity checks, scene-level experiment protocols, and model evaluation into one
pipeline. Its current Potsdam study compares complete trained systems and examines their
accuracy, class errors, scene variation, and recorded compute. It does not introduce a
new segmentation architecture or use an ensemble.

## Research focus

The study asks what accuracy and compute trade-offs the recorded systems exhibit under a
documented, scene-aware evaluation protocol on ISPRS Potsdam. It also asks where systems
agree and differ at the pixel level, and which tested training choices changed validation
results. A separate causal question—whether a particular band, augmentation, or data
preparation step improves performance under otherwise controlled conditions—remains
open. The evidence and scope of each answer are summarized in
[Research questions and contribution](docs/research.md).

## Contributions

- **Geospatial data integrity.** Scene roles, source checksums, band identity, label
  support, raster grids, and auxiliary-data alignment are recorded alongside the data
  preparation code. This includes a verified Potsdam RGB channel manifest that resolves
  legacy channel naming in the source data. See the [data profile](docs/research/potsdam-vaihingen-data-profile-2026-11-07.md)
  and [pipeline guide](docs/data-pipeline.md).
- **Experiment infrastructure.** Dataset splits, model inputs, preprocessing, sampling,
  training schedules, inference windows, and metric calculations are represented in
  versioned configurations and reports. Public confusion matrices and an evidence
  registry now let a clean clone recompute the reported Potsdam metrics without the
  authors' run directory.
- **Empirical system analysis.** The study includes DINOv2, DINOv3, PSPNet, and SAM
  single-model results, plus validation comparisons, scene-level robustness, class
  confusions, paired corrections and regressions, and measured compute. These are
  complete recipe comparisons, not architecture-only experiments.

## Results and what we learned

![SciencePlots comparison of SAM, DINOv2, DINOv3, and PSPNet on Potsdam](docs/experiments/figures/potsdam-four-model-benchmark-scienceplots-2026-11-07.png)

*The figure summarizes aggregate accuracy, class IoU, per-scene differences, and recorded
full-scene scoring time. Numerical results are recomputed in [Results](docs/results.md);
experiment and source hashes are in the [evidence registry](artifacts/public/evidence-manifest.json).*

The recorded DINOv3 RGB recipe has the highest OA and six-class mIoU among the four
systems in this historical cohort ([CLAIM-001](artifacts/public/evidence-manifest.json)).
DINOv3 RGB also scored higher than PSPNet RGB in the matched RGB evaluation setup, while
taking longer in the recorded full-scene scoring profile. This comparison covers two
complete recipes and one measured hardware setup ([CLAIM-005](artifacts/public/evidence-manifest.json)).

Three observations guide the next research steps:

1. The RGB-only DINOv3 selection recipe scored above the tested bundled auxiliary-feature
   and stronger-augmentation recipes on the recorded validation objective. The bundle
   changes multiple features at once, so it does not establish that IR, DSM, or any one
   augmentation is harmful ([CLAIM-007](artifacts/public/evidence-manifest.json)).
2. Paired prediction maps show that DINOv3 and PSPNet make both shared errors and
   model-specific corrections. Their net difference comes from the balance of these
   decisions, not from all pixels moving in one direction. The paired counts describe
   this cohort and do not identify the cause of errors ([CLAIM-006](artifacts/public/evidence-manifest.json)).
3. Low vegetation, tree, and clutter/background account for important class-level error
   differences. These patterns motivate controlled studies of class appearance and
   context; they are not causal explanations. See the [failure analysis](docs/research/failure-taxonomy-and-case-studies.md).

![Five deterministic paired DINOv3 and PSPNet error cases](docs/research/figures/paired-cases-overview.png)

*Each row shows a crop chosen by the recorded pixel-count rule for one paired error or
disagreement group. These targeted examples make the class-level patterns concrete;
they are not a random sample of Potsdam errors. Selection details and all 20 crops are
in the [failure analysis](docs/research/failure-taxonomy-and-case-studies.md).*

## Data and evaluation flow

The Potsdam split is made by scene before patches are sampled: 18 scenes for training,
six for validation, and 14 for the historical benchmark. Training uses participant
labels; validation and benchmark scoring use eroded references. The six scored classes
are impervious surfaces, building, low vegetation, tree, car, and clutter/background;
label ID 6 is ignored.

```mermaid
flowchart LR
    A[Source imagery and labels] --> B[Validate scene identity and grids]
    B --> C[Freeze split, preprocessing, and model view]
    C --> D[Train and select with validation scenes]
    D --> E[Refit selected recipe]
    E --> F[Evaluate recorded benchmark cohort]
    F --> G[Confusion matrices and error analysis]
```

The public summary uses per-scene confusion matrices and sums them before computing
aggregate metrics. This preserves pixel support and makes the calculation inspectable.
The [reproducibility guide](docs/reproducibility.md) distinguishes metric reproduction
from inference and training reproduction.

## Verify or explore

A clean clone can verify the public headline metrics using only Python's standard
library:

```bash
python3 scripts/recompute_public_metrics.py --check
```

To regenerate the numerical tables in `docs/results.md`, run
`python3 scripts/recompute_public_metrics.py --write`. The command checks registered
SHA-256 hashes, scene coverage, matrix totals, and all published aggregate values. The
[public evidence bundle](artifacts/public/README.md) contains confusion matrices and
metadata snapshots only; it does not contain imagery, labels, checkpoints, or prediction
rasters.

Run `python3 scripts/check_public_docs.py` to verify local Markdown links, evidence IDs,
experiment status markers, and the generated results section.

For implementation details, read [architecture](docs/architecture.md), [data pipeline](docs/data-pipeline.md),
and the [experiment index](docs/experiments/README.md). The index marks current records,
historical reports, superseded phase notes, and designs that have not been run. Figure
sources and reproduction requirements are listed in [Figures](docs/figures.md).

## Scope and limitations

The 14-scene Potsdam cohort has been used in earlier campaigns. It is a historical
benchmark, not an untouched test set or independent evidence of geographic
generalization. The comparison has one final checkpoint per recipe, and millions of
pixels are spatially dependent. Eroded labels exclude boundaries. Runtime reflects
specific archived Kaggle profiles. See [Limitations](docs/limitations.md) for internal,
external, statistical, label, compute, and artifact scope.

The repository has no code license declaration. Dataset imagery and pretrained weights
have separate source terms; this repository does not redistribute them. Reuse and
redistribution rights are not granted by the public metrics bundle.
