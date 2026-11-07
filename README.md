# DeepNet: semantic segmentation of aerial imagery

**Research question.** How far can a scene-aware, auditable training pipeline improve
semantic segmentation of high-resolution Potsdam orthophotos, and what accuracy–runtime
trade-off does it show against a conventional convolutional model?

**Current thesis.** On the 14-scene Potsdam benchmark cohort, the DINOv3-B RGB recipe is
the strongest evaluated system so far (92.04% overall accuracy and 81.09% six-class
mIoU). The matched PSPNet RGB recipe scores slightly lower but takes less time to score
the full scenes. Neither reaches the 99% OA target. The largest remaining weakness is
clutter/background, while low-vegetation and tree confusions also account for many
errors. Adding the bundled IR/index/DSM features or stronger generic augmentation did
not improve the DINOv3 selection result. These findings describe complete model recipes
on a benchmark cohort whose scores were seen in earlier campaigns; they do not establish
performance on a newly sampled city or untouched test set.

## The result in one view

The figure puts four Potsdam systems on the same historical 14-scene, 467,862,855-pixel
cohort. It brings together aggregate accuracy, class-level IoU, scene-to-scene
differences, and the recorded full-scene scoring time.

![SciencePlots comparison of SAM, DINOv2, DINOv3, and PSPNet on the Potsdam benchmark](docs/experiments/figures/potsdam-four-model-benchmark-scienceplots-2026-11-07.png)

*Figure 1. Aggregate OA and six-class mIoU; IoU for impervious surfaces, buildings, low
vegetation, trees, cars, and clutter/background; per-scene OA relative to DINOv3; and
recorded full-scene scoring time, rather than model-forward-only time. The 99% reference
applies to OA. PSPNet and DINOv3 share the RGB cache and 512×512 windows at stride 256.
SAM and DINOv2 retain their historical inference recipes, so this is a recipe-level
comparison rather than a controlled architecture-only study. The plotted data and figure
command are documented in [Figure
reproduction](docs/figures.md#consolidated-readme-benchmark).*

| System | OA | Six-class mIoU | Incorrect pixels |
| --- | ---: | ---: | ---: |
| SAM ViT-B + LoRA | 87.8142% | 73.7024% | 57,012,689 |
| DINOv2 refit | 91.4785% | 79.6122% | 39,868,882 |
| PSPNet–ResNet101 RGB refit | 91.2834% | 79.4235% | 40,781,843 |
| **DINOv3-B RGB refit** | **92.0397%** | **81.0932%** | **37,243,220** |

The DINOv3 refit improves on DINOv2 by 0.5612 percentage points OA and 1.4810 mIoU
points, reducing errors by 2,625,662 (6.59%). It improves OA on 12 of 14 scenes and
raises IoU on five of six classes. Clutter IoU rises by 6.0635 points to 48.4383%; car
IoU changes by −0.2816 point to 92.4418%. Even so, DINOv3 is 6.9603 OA points below 99%
and would need 32,564,592 fewer errors on this cohort—an 87.44% reduction from its
current error count. The [DINOv3 final
review](docs/experiments/potsdam-dinov3-rgbir-v2-final-review-2026-11-06.md) contains
the confusion matrices, per-scene scores, checkpoint audit, and limits of this
comparison.

PSPNet trails DINOv3 by 0.7563 OA points and 1.6697 mIoU points, with lower IoU in all
six classes and lower OA on 13 of 14 scenes. Its recorded full-scene scoring time is
254.83 seconds versus 405.23 seconds for DINOv3. Across their matched selection, refit,
and benchmark phases, the campaigns used 1.9066 and 3.8218 allocated GPU-hours
respectively. The time and GPU-hour results cover those recorded jobs and do not
establish a general hardware-independent speed ratio. The [PSPNet
report](docs/experiments/potsdam-pspnet-rgb-v1.md) documents its single-seed recipe,
phase budgets, and benchmark audit.

## How the evidence led here

### 1. Validate the imagery before changing the model

Potsdam's official RGBIR source is ordered red, green, blue, infrared. A source audit
found that the historical three-band files correspond to RGBIR bands 4/2/3—near
infrared, green, blue—even though older configurations named them IR/red/green. The
winning DINOv3 encoder instead receives verified RGB bands 1/2/3. Vaihingen's official
three-band order is near infrared, red, green. The full [Potsdam and Vaihingen data
profile](docs/research/potsdam-vaihingen-data-profile-2026-11-07.md) records these
mappings, current scene coverage, label composition, optical distributions, and DSM
checks. Digital numbers are not calibrated reflectance.

All training and evaluation samples are assigned by scene before patches are drawn.
Dataset catalogs bind source files to hashes, band roles, and raster grids; split
manifests bind scene IDs to label roles. The DINOv3 campaign compared memory-mapped
caches against Rasterio reads before use. Its winning A recipe uses RGB only.
Configuration B bundled IR, a post-augmentation spectral index, and verified DSM relief,
but did not beat A; because those signals were grouped, the result does not isolate
their individual effects. JPEG nDSM remains excluded because registration and height
encoding have not been established. See the [data pipeline
guide](docs/data-pipeline.md), [DSM snapshot](data/auxiliary/potsdam-dsm-v1.dvc), and
[model architecture](docs/architecture.md).

### 2. Select a recipe on validation scenes, then refit

The Potsdam project split has 18 train, 6 validation, and 14 benchmark scenes. During
selection, participant labels train on the first 18 scenes; the six validation scenes
use eroded labels and remain reserved for model selection. After selection and seed
confirmation, the winning checkpoint continues for exactly 1,152 updates on participant
labels from all 24 train-plus-validation scenes. Only that final refit checkpoint opens
the benchmark evaluation. This order prevents the refit from using the 14 benchmark
scenes.

The [DINOv2 coverage study](docs/experiments/potsdam-dinov2-coverage-refit.md)
established the earlier pretrained baseline. Its winning C recipe used ViT-B/14 with the
historical IRRG input, selected on six validation scenes, then reached 91.4785% OA /
79.6122% mIoU on the benchmark after refit. The [artifact
review](docs/experiments/potsdam-dinov2-coverage-refit-final-review-2026-11-05.md)
contains its refit and scoring audit. The DINOv3 campaign then tested pinned ViT-B/16
with verified RGB against a larger satellite-pretrained ViT-L pilot. ViT-B fit the
campaign budget; ViT-L exceeded its cost limit. On validation, RGB-only A won over B
with bundled auxiliary features and C with stronger augmentation. Seed 43 confirmed A,
and horizontal-flip test-time augmentation did not meet its predeclared gain threshold.

| DINOv3 selection arm | Input and augmentation | Validation OA | Validation mIoU |
| --- | --- | ---: | ---: |
| A | RGB, baseline augmentation | **93.0441%** | **80.7498%** |
| B | A plus IR, index, and DSM features | 92.9923% | 80.6097% |
| C | A with stronger augmentation | 92.9690% | 80.5115% |
| Seed-43 confirmation | A recipe, independent seed | 92.9477% | 80.4596% |

Selection scores use six scenes and 199,409,714 scored pixels. The validation chart and
full decision rule are in the [DINOv3 selection
review](docs/experiments/potsdam-dinov3-rgbir-v2-selection-summary-2026-11-06.md); the
[preflight report](docs/experiments/potsdam-dinov3-rgbir-v2-preflight-2026-11-05.md)
records the real-backbone, small-fit, and throughput checks. The RGB-only result is
evidence about these tested recipes, not proof that infrared or elevation cannot help
under other fusion designs.

### 3. Compare complete systems on the fixed cohort

All four rows in Figure 1 use the same 14 scene IDs, six class supports, and 467,862,855
scored pixels. Validation and benchmark scoring use eroded references; participant
labels are used for training. Aggregate OA, mIoU, class IoU, and F1 are recomputed from
confusion matrices summed across scenes. The saved reviews confirm matching scene IDs,
counts, and class supports; byte-for-byte equality of the reference rasters in the older
SAM and DINOv2 runs was not re-established. The 14 benchmark scenes were evaluated in
previous campaigns, so the scores are repeatable historical comparisons rather than a
fresh blind test.

The DINOv3-B RGB model uses pinned pretrained weights, LoRA on attention projections,
pretrained normalization at blocks 3/6/9/12, and a width-128 GroupNorm decoder with a
stride-four detail branch. Its [model spec](configs/models/dinov3-vitb16-rgbir-v2.json)
binds the named RGB inputs and training contract. The DINOv2 result uses a different
encoder size, channel recipe, and inference tiling; the OA change cannot be attributed
to the encoder version alone. PSPNet uses ImageNet-pretrained ResNet101 with pyramid
pooling and a stride-four detail path. The [architecture guide](docs/architecture.md)
and [PSPNet comparison report](docs/experiments/potsdam-pspnet-rgb-v1.md) describe the
other model and protocol details. PSPNet is one seed; its result compares complete
systems and does not isolate the backbone.

The 99% target is still an open research goal. The current matrices place the most
direct next questions around clutter/background, class-boundary ambiguity, and
low-vegetation/tree confusions. The next improvement should be tested with per-class
evidence and a held-out region not previously used to compare these recipes.

## Data and evaluation contract

| Dataset | Scene roles | Training labels | Validation and benchmark labels |
| --- | --- | --- | --- |
| [ISPRS Potsdam](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx) | 18 train / 6 validation / 14 benchmark | Participant labels | Eroded labels |
| [ISPRS Vaihingen](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx) | 12 train / 4 validation / 17 benchmark | Participant labels | Eroded labels |

These are project-defined scene splits. Each benchmark metric covers six classes:
impervious surfaces, buildings, low vegetation, trees, cars, and clutter/background.
Label ID 6 is ignored. OA is the share of valid pixels classified correctly; mIoU is the
unweighted mean of the six class IoUs. Reported per-scene metrics are reduced from each
scene's confusion matrix, and aggregate scores are recomputed after summing matrices.
The [ISPRS Vaihingen
page](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx)
and [data profile](docs/research/potsdam-vaihingen-data-profile-2026-11-07.md) describe
its resolution and channel order.

The older DINOv2 runs and the current RGB experiments do not use identical complete
pipelines: DINOv2 uses 504×504 windows at stride 252, while PSPNet and DINOv3 use
512×512 windows at stride 256. SAM retains its historical tiling. The DINOv3–PSPNet run
shares the cache, RGB inputs, sampling schedule, and scoring geometry. The figure
therefore supports an audited recipe comparison and a matched PSPNet–DINOv3 comparison,
not a claim that architecture alone caused the differences.

After refit, held-out scoring is gated on the final checkpoint and run manifest:

```mermaid
flowchart LR
    A[18 train + 6 validation scenes] --> B[Select recipe and confirm seed]
    B --> C[Refit winning checkpoint on 24 participant-labeled scenes]
    C --> D[Score final checkpoint once on 14 benchmark scenes]
```

The implementation keeps train/validation bundles separate from the test-only evaluation
pipeline. Cache and run manifests bind hashes, preprocessing, label roles, model inputs,
checkpoints, and metrics. Persisted confusion matrices let reviews recompute scores
without running model inference.

## Reproduce and explore

To recreate the README comparison figure from the saved audited run outputs, install the
locked notebook extra and run the renderer. The external run roots contain the metric
artifacts; training images, caches, checkpoints, and logs are not stored in Git.

```bash
uv sync --locked --extra notebooks
uv run --locked --extra notebooks python -m scripts.render_scienceplots model-comparison \
  /path/to/runs/potsdam-pspnet-rgb-v1 \
  docs/experiments/figures/potsdam-four-model-benchmark-scienceplots-2026-11-07.png \
  --dinov3-run /path/to/runs/potsdam-dinov3-rgbir-v2
```

For data preparation and new experiments, start with the [DINOv3 RGBIR
protocol](docs/experiments/potsdam-dinov3-rgbir-v2.md), [data pipeline
guide](docs/data-pipeline.md), and [evaluation DVC
pipeline](pipelines/evaluation/dvc.yaml). DINOv3 pretrained weights require gated
Hugging Face access. The Kaggle selection, refit, and one-time benchmark jobs are linked
in the [DINOv3 protocol](docs/experiments/potsdam-dinov3-rgbir-v2.md); the separate
[PSPNet runbook and result](docs/experiments/potsdam-pspnet-rgb-v1.md) reproduce that
comparison. Figure commands for DINOv3 selection, EDA, and legacy notebooks are in
[Figure reproduction](docs/figures.md).

| Project area | Purpose |
| --- | --- |
| [`configs/`](configs/) | Dataset splits, inputs, model specs, and experiment protocols |
| [`deepnet/data/`](deepnet/data/) | Raster catalog, bundles, sampling, and evaluation |
| [`deepnet/models/`](deepnet/models/) | Pretrained encoders, adapters, and segmentation decoders |
| [`deepnet/potsdam_rgbir/`](deepnet/potsdam_rgbir/) | DINOv3 RGBIR cache, model checks, training phases, and protocol |
| [`notebooks_src/`](notebooks_src/) | EDA and augmentation analyses paired with notebooks |
| [`docs/experiments/`](docs/experiments/) | Frozen experiment protocols, results, and artifact reviews |
| [`docs/research/`](docs/research/) | Potsdam/Vaihingen data and channel analysis |

The archived joint DINOv2 study also evaluated Potsdam and Vaihingen. It reported 74.66%
OA / 48.65% mIoU on Potsdam and 78.30% / 49.18% on Vaihingen, but used joint training
and different inference recipes. Its 14 Potsdam and 17 Vaihingen scene results are
reported separately in the [joint DINOv2 final
review](docs/experiments/isprs-dinov2-joint-final-review-2026-11-01.md); do not rank
those numbers directly against Figure 1.
