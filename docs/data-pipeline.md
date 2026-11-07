# Data pipeline guide

Build catalogs and bundles from source rasters for local or Kaggle runs. Keep
rasters and run outputs outside Git.

## Completed Potsdam DINOv3 campaign

The [Potsdam RGBIR runbook](experiments/potsdam-dinov3-rgbir-v2.md) covers
selection, refit and scoring. See the [final
review](experiments/potsdam-dinov3-rgbir-v2-final-review-2026-11-06.md) for
results, timing and saved artifacts.

| Phase | Data scope | Checkpoint rule |
| --- | --- | --- |
| Selection and confirmation | 18 train scenes with participant labels; six validation scenes with eroded labels and 199,409,714 scored pixels | A/B/C start from pretrained with seed 42; D confirms the winner from pretrained with seed 43; full validation at passes 4/8 selects checkpoints |
| Refit | Separate train-only cache containing the 24-scene union, participant labels only | Continue the locked seed-42 winner for exactly 1,152 updates with reset optimizer/scaler; all reported metrics are train metrics |
| Final benchmark | Separate 14-scene eroded-label cache; 467,862,855 scored pixels | Score the completed refit checkpoint once with the frozen policy; review stored matrices afterward |

The RGBIR snapshot uses R/G/B/IR order. The encoder reads RGB. The optional
branch combines IR, NDVI computed after augmentation, DSM relief scaled on the
training scenes, and an availability mask. Selection chose RGB-only A with
baseline augmentation. The benchmark uses identity inference with **512 × 512
windows and stride 256**.

`deepnet.potsdam_rgbir.prepare` builds memory-mapped arrays and records source
hashes, split, preprocessing, feature statistics and label roles. Verify cached
arrays, label histograms and scored-pixel counts against Rasterio reads. The
image/label mask determines scoring support, including where scene `3_13` lacks
DSM. Relief normalization uses the original 18 training scenes.

Use `deepnet.potsdam_rgbir.study` and the Kaggle coordinator with the separate
RGBIR snapshot and caches. To review a completed run, use its saved confusion
matrices.

## Completed Potsdam PSPNet comparison

The [PSPNet RGB runbook and results](experiments/potsdam-pspnet-rgb-v1.md) document
a single-seed comparison that reuses the winning DINOv3 recipe's role-scoped
caches, sampler and baseline augmentation. PSPNet selects its own checkpoint
on six validation scenes, refits for 1,152 updates on the 24-scene train-only
cache, then scores that final checkpoint once on the same 14-scene benchmark.
It reaches **91.2834% OA / 79.4235% mIoU** on **467,862,855 valid pixels**.
Use the saved confusion matrices to review these results.

## Install

The package requires Python 3.10 or newer. For local development, create an
isolated environment and install the data, training, model, augmentation, and
lint/test dependencies:

```bash
uv sync --extra dev --extra train --extra models --extra vision --extra data-versioning --extra notebooks
```

Check the interpreter, Torch version and device after installation. On Apple
Silicon, the runtime selects MPS when Torch exposes it:

```bash
python -c 'from deepnet.runtime import configure_torch_runtime, runtime_metadata; print(runtime_metadata(configure_torch_runtime()))'
```

`rasterio` is required for raster access. PyTorch remains an optional dependency of
the package itself. On Kaggle, use the preinstalled Torch build for its configured
accelerator and install the project wheel without resolving dependencies:

```bash
python -m pip wheel --no-deps --wheel-dir /path/to/artifacts .
```

Upload the resulting pure-Python `.whl` as a private Kaggle Dataset, attach it to
the notebook, then install it from the read-only input mount:

```python
%pip install --no-deps /kaggle/input/deepnet-wheel/deepnet_satellite_labeling-0.1.0-py3-none-any.whl
```

The wheel contains the CLI and default JSON contracts. Check that `numpy`,
`rasterio` and `torch` import. Install a compatible Rasterio wheel if needed.

For DINOv2/DINOv3 or domain-specific augmentation, install the optional packages in
the active Kaggle environment while retaining its existing Torch package:

```python
%pip install "albumentations==2.0.8" "transformers>=4.57,<6" "peft>=0.14,<1"
```

The configured backbone loader fetches pretrained weights on demand.

## Create and validate a scene catalog

After attaching the GPU accelerator and installing the project wheel, verify that
the active kernel sees CUDA before loading a bundle:

```python
from deepnet.runtime import configure_torch_runtime, runtime_metadata

runtime = configure_torch_runtime()
if runtime.device.type != "cuda":
    raise RuntimeError("Enable a Kaggle GPU accelerator before starting training")
print(runtime_metadata(runtime))
```

Choose a data root containing the relative paths in the dataset descriptor.

```bash
export DATA_ROOT="/path/to/inputs"

deepnet-data build-catalog \
  --dataset-id isprs-potsdam \
  --data-root "$DATA_ROOT" \
  --output "$DATA_ROOT/manifests/potsdam-catalog.json" \
  --hash-assets

deepnet-data validate-catalog \
  --catalog "$DATA_ROOT/manifests/potsdam-catalog.json" \
  --split configs/splits/potsdam-legacy-v1.json \
  --view-assets irrg-view=irrg
```

Catalog creation reads source headers and optionally hashes each file. Rebuild
the catalog as a new version when the source inventory or metadata changes.

Generate a windowed quality report before creating a training bundle:

```bash
deepnet-data quality-report \
  --catalog "$DATA_ROOT/manifests/potsdam-catalog.json" \
  --split configs/splits/potsdam-legacy-v1.json \
  --preprocessing-id isprs-canonical-v1 \
  --data-root "$DATA_ROOT" \
  --tile-size 1024 \
  --view-assets irrg-view=irrg \
  --output "$DATA_ROOT/manifests/potsdam-quality-v1.json"
```

Each required view is checked against the per-scene reference grid and the label
asset selected by the split. The scan opens every required image, verifies its
metadata, and intersects its raster mask and finite-value mask with the label mask.
Missing images, changed metadata, unsupported alignment, and unknown valid label
colors outside the configured palette tolerance fail explicitly. Bundle verification
checks content hashes separately.

Each scene and split reports coverage by view. `valid_pixels` counts the
intersection of all required views. `class_counts` includes ignore labels;
`scored_pixels` excludes them.

## Explore the data and review augmentation

Set `DEEPNET_DATA_ROOT` to the materialized inputs. The notebook uses catalogs
and quality reports for train and validation scenes. It saves asset coverage,
alignment checks, label counts, IRRG histograms, elevation checks and
augmentation examples under `DEEPNET_ARTIFACT_ROOT`.

Open the runnable
[`notebooks/data_eda_augmentation.ipynb`](../notebooks/data_eda_augmentation.ipynb)
after installing `.[notebooks,vision,data-versioning]`, materializing the DVC inputs,
and running `dvc repro`. Its companion Jupytext source is
[`notebooks_src/data_eda_augmentation.py`](../notebooks_src/data_eda_augmentation.py).
Register the project environment once and select **Python (DeepNet EDA)** in
Jupyter:

```bash
uv run python -m ipykernel install --user \
  --name deepnet-eda --display-name "Python (DeepNet EDA)"
```

The committed notebook contains source cells without generated outputs. Run all
cells from the top; tables, figures and provenance go to `DEEPNET_ARTIFACT_ROOT`.
To redraw the charts from existing CSVs without rescanning rasters, see
[Reproduce the figures](figures.md#eda-and-legacy-notebooks). To regenerate the
notebook from its source, run:

```bash
uv run jupytext --to ipynb \
  --output notebooks/data_eda_augmentation.ipynb \
  notebooks_src/data_eda_augmentation.py
```

Set `DEEPNET_DATA_ROOT` to `data/raw/inputs` and `DEEPNET_ARTIFACT_ROOT` to a
directory outside the repository. Potsdam uses eroded labels for its six
validation scenes; Vaihingen uses eroded labels for its four validation scenes.
Training reads participant labels.

The current canonical preprocessing v2 retains the official ISPRS RGB palette and
allows a maximum 80-point Euclidean RGB distance for source color-managed labels.
This source-specific need was confirmed by comparing Potsdam scene 4_12's full
noBoundary label to its exact-palette participant noBoundary counterpart: every
non-ignore pixel decoded to the same class, with a maximum nearest-palette distance
of 69.88. Tied colors or colors farther than the configured tolerance still fail.
The official six-class palette and ignored black boundaries are documented by
[ISPRS](https://isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx).

### Unexecuted joint DINOv3 split and evaluation protocol

The completed Potsdam campaigns share these Potsdam scene assignments. See the
[DINOv3](experiments/potsdam-dinov3-rgbir-v2.md) and
[DINOv2](experiments/potsdam-dinov2-coverage-refit.md) runbooks for their
selection, refit and evaluation rules.

The joint plan follows the [ISPRS
contest](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx)
partition of 24 participant-labeled tiles and 14 challenge tiles, as described
in the [2015 ISPRS
report](https://isprs.org/society/si/SI-2015/ISPRS-SI2015-Gerke_report_WGIII4_2015.pdf).
The project fixes an internal 18/6 split of the 24 labeled tiles, using the same
counts as a later [ISPRS Annals
study](https://isprs-annals.copernicus.org/articles/V-2-2022/275/2022/isprs-annals-V-2-2022-275-2022.pdf).

The joint [evaluation protocol](../configs/evaluations/isprs-dinov3-v2.json)
binds the training plan and split manifests by SHA-256: [Potsdam
v2](../configs/splits/potsdam-dinov3-v2.json) and [Vaihingen
v1](../configs/splits/vaihingen-dinov3-v1.json).

Exact scene membership is:

- **Potsdam train (18):** `2_10`, `2_11`, `2_12`, `3_10`, `3_11`, `5_10`, `5_12`,
  `6_10`, `6_11`, `6_7`, `6_8`, `6_9`, `7_10`, `7_11`, `7_12`, `7_7`, `7_8`, `7_9`.
- **Potsdam internal validation (6):** `3_12`, `4_10`, `4_11`, `4_12`, `5_11`, `6_12`.
- **Potsdam challenge test (14):** `2_13`, `2_14`, `3_13`, `3_14`, `4_13`, `4_14`,
  `4_15`, `5_13`, `5_14`, `5_15`, `6_13`, `6_14`, `6_15`, `7_13`.
- **Vaihingen train (12):** `3`, `5`, `13`, `15`, `17`, `21`, `23`, `26`, `30`,
  `32`, `34`, `37`.
- **Vaihingen validation (4):** `1`, `7`, `11`, `28`.
- **Vaihingen post-selection test (17):** `2`, `4`, `6`, `8`, `10`, `12`, `14`,
  `16`, `20`, `22`, `24`, `27`, `29`, `31`, `33`, `35`, `38`.

Select checkpoints by the unweighted mean of Potsdam and Vaihingen validation
mIoU. Freeze the recipe and checkpoint in the run manifest, then score the 14
Potsdam and 17 Vaihingen test scenes once. The default DVC pipeline exports
train and validation bundles.

Vaihingen test scores would measure performance on held-out scenes after joint
training. To measure transfer to Vaihingen, train and select on Potsdam alone.

The separate [evaluation DVC pipeline](../pipelines/evaluation/dvc.yaml)
validates the test splits, scans image/label quality and exports test-only
bundles. Run each target explicitly:

```bash
uv run dvc repro pipelines/evaluation/dvc.yaml:bundle_potsdam_test
uv run dvc repro pipelines/evaluation/dvc.yaml:bundle_vaihingen_test
```

DVC tracks eroded references for the 14 Potsdam test scenes and four validation
plus 17 test scenes in Vaihingen. For each dataset, the evaluation pipeline
writes `post-selection-test-validation.json`, `post-selection-test-quality.json`
and a test-only bundle. Mount those bundles only for evaluation of the frozen
checkpoint.

The [ISPRS Potsdam results
table](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/results/label_challenge_2D_potsdam_result_ifr.html)
reports five class F1 scores (excluding clutter/background) and OA against
eroded references. The local plan computes those metrics and six-class F1/IoU,
precision, recall and mIoU by summing per-scene confusion matrices. The catalog
has eroded references for these test tiles; equivalence to the organizers'
scoring files is unverified.

[`evaluate_scenes`](../deepnet/data/evaluation.py) runs deterministic tiled
inference, blends overlapping logits and scores per-scene and aggregate
confusion matrices. The joint plan uses `deepnet.training`. The Potsdam campaign
uses `deepnet.potsdam_rgbir.study`, which requires the final refit checkpoint.
Both restrict evaluation to the selected manifest.

By default, stride is `floor(window_size / 2)` per axis, with a minimum of one
pixel. A custom stride must be smaller than the window on axes where the window
exceeds one pixel. The evaluation records the resolved stride and window size.

The 14 Potsdam test scenes have been scored in prior experiments. The splits
separate scenes; neighboring tiles can share geographic context.

### Historical EDA snapshot from DINOv3 v1

The v1 snapshot uses an 18/20 Potsdam split that included 14 challenge scenes in
validation. Regenerate the EDA with the v2 18/6/14 split for current validation
statistics. The augmentation replay used the same 18 training scenes as v2.

The previous analysis was generated against DVC lock SHA-256
`17994ddb59a804bc847d3429cdd17dce1b0435db1c054e207e15565f28217b9c`. Label
proportions below come from complete train/validation scene scans. Each class share
is normalized over the six semantic classes; the ignore share uses all labeled
pixels.

| Dataset / role | Impervious | Building | Low vegetation | Tree | Car | Clutter | Ignore |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Potsdam train (18 scenes) | 28.47% | 24.88% | 24.78% | 14.64% | 1.61% | 5.61% | 0.00% |
| Potsdam validation (20 scenes) | 30.57% | 27.58% | 20.21% | 16.33% | 1.58% | 3.74% | 7.32% |
| Vaihingen train (12 scenes) | 26.84% | 25.20% | 22.20% | 23.82% | 1.06% | 0.88% | 0.00% |
| Vaihingen validation (4 scenes) | 31.18% | 30.09% | 16.07% | 21.40% | 1.23% | 0.03% | 9.94% |

Cars account for about 1% of semantic pixels. Vaihingen clutter appears in 4/12
train and 1/4 validation scenes. Training uses a 30% class-aware component;
validation uses the natural class distribution.

The 2026-10-22 v1 profile scanned 38 Potsdam and 16 Vaihingen scenes in native
raster blocks. Histograms and quantiles are exact for uint8 values. All scanned
IRRG pixels passed the mask and finite-value checks.

The historical v1 table reports pooled band means and medians plus the range of
per-scene medians in the configured IR, red, green order. Its per-scene CSV retains
exact minima, percentiles, standard deviations, valid-pixel counts, and full
256-bin histograms.

| Dataset / role | Scenes / pixels per band | Mean IR / red / green | Pooled median IR / red / green | Per-scene median range IR / red / green |
| --- | ---: | --- | --- | --- |
| Potsdam train | 18 / 648,000,000 | 98.71 / 92.57 / 85.55 | 95 / 85 / 77 | 79–119 / 78–91 / 71–86 |
| Potsdam validation | 20 / 720,000,000 | 96.68 / 92.52 / 86.25 | 92 / 86 / 79 | 80–105 / 78–94 / 71–88 |
| Vaihingen train | 12 / 58,554,459 | 121.39 / 82.01 / 80.86 | 121 / 72 / 76 | 98–150 / 58–101 / 61–97 |
| Vaihingen validation | 4 / 19,533,347 | 114.82 / 79.61 / 78.70 | 112 / 71 / 74 | 101–118 / 64–73 / 70–76 |

Column names follow the v1 configs. In Potsdam, the source audit mapped the
three bands to RGBIR 4/2/3 (IR/G/B). For Potsdam, the second and third band
statistics measure green and blue. The audit covered Potsdam only.

Vaihingen's IR channel is brighter and varies more across scenes than Potsdam's.
See `irrg-band-statistics.csv`, `irrg-per-scene-band-statistics.csv` and
`irrg-full-histograms.csv` in the EDA output directory.

All 54 image/label pairs passed alignment checks. Potsdam IRRG has CRS and
affine metadata; its pixel-local labels use source-attested scene pairs.
Vaihingen imagery and labels are pixel-local and use explicit ISPRS pair
records.

### Potsdam DSM/nDSM alignment and value audit

The historical notebook audited 38 Potsdam DSM TIFFs and 38 configured lastools
nDSM files from `DEEPNET_AUX_DATA_ROOT`. None was in the raw DVC input snapshot
at the time. The [separate DSM snapshot](../data/auxiliary/potsdam-dsm-v1.dvc)
now tracks the verified source TIFFs and world files. The Potsdam coverage study
derives local relief from them and uses an availability mask for the missing
column in scene `3_13`. Its DSM branch is separate from the legacy DSM+nDSM
model view described below; JPEG nDSM remains excluded.

| Asset | Result against the matching IRRG reference grid | Scene match vs pixel-grid evidence |
| --- | --- | --- |
| DSM GeoTIFF | 37/38 have matching EPSG:32633 CRS, affine transform, width, and height (6000 x 6000). | Scene `3_13` shares CRS and affine origin/resolution but measures 5999 x 6000, with one 6000-pixel column missing. |
| nDSM JPEG | 37/38 have the same 6000 x 6000 dimensions; `3_13` is 5999 x 6000. | The files lack CRS and affine metadata. Co-registration is unverified. |

The DSM TIFFs have no embedded vertical-unit metadata; the dataset config describes
them as metres, but the source unit and vertical datum still need authoritative
confirmation. Full scans found DSM values from -35.75 in train scene `7_10` to
106.171 in scene `7_13` (validation in historical v1; benchmark in the current
split). These extrema need source review before choosing elevation clipping or
normalization. DSM per-scene mean/min/max and standard
deviation use all valid pixels; its p01–p99 values use a deterministic,
block-stratified sample of about 100,000 pixels per scene. The nDSM JPEG stores
uint8 digital numbers from 0 to 255; its histograms and quantiles are exact encoded
value summaries, not metric heights.

Before using the legacy five-channel view, obtain authoritative nDSM decoding
and co-registration evidence and version that source with checksums. Verify the
full grids or document a supported missing-data mask, then fit elevation
statistics on training scenes only. The unexecuted joint DINOv3 configuration uses
the historical IRRG view only; the completed RGBIR campaign has its own verified
RGB/auxiliary routing.

### Augmentation policy and audit

`isprs-irrg-aerial-v2` uses all square symmetries and a 35% affine transform with
scale 0.9–1.1, rotation ±12 degrees, translation ±4%, linear image interpolation,
nearest-neighbor labels, and ignore-filled borders. Brightness/contrast, gamma,
Gaussian noise, and 3–5 pixel blur are independently sampled with probabilities
35%, 20%, 15%, and 10%. Photometric transforms operate only on IRRG and leave
labels, validity masks, and auxiliary bands unchanged. The IRRG values are
multispectral, so an RGB hue/saturation transform is intentionally excluded.

The augmenter gives each transform a separate seed. The pinned Albumentations
`Compose.set_random_seed()` assigns the same seed to every child, correlating
their apply/skip decisions. The regression test checks all four combinations
for two 50% transforms and exact replay with the same sample seed.

The audit samples one training crop from each train scene and replays it with three
seeds (54 Potsdam and 36 Vaihingen samples). Every source class present in these
sampled windows remained represented after geometry. The fifth-percentile valid
area was 88.9% for Potsdam and 88.8% for Vaihingen; median valid area was 100% in
both. Conditional on a photometric change, the median absolute change was 5.9–6.8
DN per Potsdam band and 3.8–4.3 DN per Vaihingen band.

Review `augmentation-examples.png`, `augmentation-impact.png`,
`augmentation-audit.csv`, and `augmentation-summary.csv` before changing the
versioned policy. `eda-provenance.json` records the source hash, DVC lock, catalogs,
splits, quality reports, runtime versions, and seeds.

#### Rare classes, semantic boundaries, and padding follow-up

The audit scanned 12,121 overlapping windows at half-window stride on training
scenes (9,522 from 18 Potsdam scenes; 2,599 from 12 Vaihingen scenes). A class
is rare if it occupies at most 6% of training pixels or appears in at most half
the training scenes. This selects car and clutter/background in both datasets.

For each dataset, the audit selected five crops: the highest-density crop and the
crop nearest the 10th percentile of positive-window density for each rare class,
plus one distinct boundary-rich crop. Each crop was replayed with eight deterministic
seeds (40 trials per dataset). The table reports geometry-only class-pixel retention;
the 5th percentile and disappearance rate are computed over trials where that
source class was present. Class-edge fractions count valid 4-neighbor semantic
transitions on both sides of the edge. For each row, the fraction is boundary pixels
of the target class divided by all valid pixels of that class in the crop; values are
averaged over seeds.

| Dataset / class | Train pixel share | Class crops / trials | Median retention | 5th percentile | Disappeared | Class-edge fraction, source → after geometry |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Potsdam car | 1.61% | 2 / 16 | 100% | 0% | 12.5% | 10.51% → 10.20% |
| Potsdam clutter/background | 5.61% | 4 / 32 | 100% | 10.2% | 0% | 7.28% → 10.87% |
| Vaihingen car | 1.06% | 2 / 16 | 100% | 73.9% | 0% | 14.13% → 14.65% |
| Vaihingen clutter/background | 0.88% | 2 / 16 | 100% | 92.0% | 0% | 4.47% → 4.60% |

Both car disappearances occurred in affine replays of the low-density crop in
Potsdam train scene `6_7` at `(row=4608, col=3584)`, which contained 1,274 car
pixels. The 4-neighbor measure counts semantic boundaries between classes.

| Dataset | Trials with affine padding | Mean / p95 / max padded area | Padding labels ignore and mask invalid | Nonzero padded values, after geometry → after radiometry | Trials where radiometry changed padding | Mean absolute change per padded pixel | Model patch padding |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Potsdam | 45% | 2.95% / 9.89% / 13.50% | 100% / 100% | 2.93% → 27.08% | 30% | 2.84 DN | 0% |
| Vaihingen | 40% | 3.02% / 13.06% / 14.56% | 100% / 100% | 8.23% → 33.33% | 25% | 4.33 DN | 0% |

Padding is labeled and masked correctly, but photometric transforms change the image
values in those invalid pixels. The nonzero percentages and mean absolute changes
are averaged over trials with affine padding; the changed-trial percentage uses all
replay trials. Model patch padding is zero because both crop sizes (512 and 288) are
divisible by DINOv3 ViT-B/16's patch size of 16.

On 2026-10-31, the augmenter was corrected to preserve pre-radiometry values at
invalid pixels. `isprs-irrg-aerial-v3` uses scale 0.95–1.05, rotation ±6°,
translation ±2% and affine probability 0.25, with the v2 optical transforms.
Replaying the train-only crops at DINOv2's 504/280 window sizes gave 80 trials,
including 17 with invalid pixels. In these trials the fill stayed unchanged and
every source-present car/clutter class survived. Reproduce with
`scripts/audit_training_augmentation.py` and the archived crop CSV.

The expanded artifacts are stored outside Git under
`/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/eda/20261022T063946Z/`:
`augmentation-training-window-scan.csv`, `augmentation-rare-boundary-manifest.csv`,
`augmentation-rare-boundary-audit.csv`, `augmentation-rare-boundary-summary.csv`,
`augmentation-padding-audit.csv`, `augmentation-padding-summary.csv`, and
`augmentation-rare-boundary-examples.png`. In the figure, cyan marks semantic class
edges and magenta marks affine-generated padding. `eda-provenance.json` includes the
split/catalog/config hashes, seeds, and runtime details.

Of 33 Vaihingen image scenes, 17 lack participant labels. Select labeled scenes
for supervised splits; use the image-only catalog for inference.

## Fit train-only statistics and export a model view

`isprs-irrg-zscore-v1` demonstrates train-only fitting on aligned IRRG bands. The
same command accepts any configured view; elevation-channel fitting also needs a
verified nDSM alignment record:

```bash
mkdir -p "$DATA_ROOT/manifests/preprocessing-v2"
deepnet-data fit-statistics \
  --catalog "$DATA_ROOT/manifests/potsdam-catalog.json" \
  --split configs/splits/potsdam-legacy-v1.json \
  --preprocessing-id isprs-canonical-v1 \
  --model-input-id isprs-irrg-zscore-v1 \
  --data-root "$DATA_ROOT" \
  --tile-size 1024 \
  --output "$DATA_ROOT/manifests/preprocessing-v2/isprs-canonical.json"
```

The result includes the exact training scene IDs, split hash, statistics, and a
new preprocessing version. Pass the directory containing that JSON to bundle
export. When multiple model views fit different channels, save each resulting spec
as an immutable preprocessing artifact and retain all fitted channels for that
split.

```bash
deepnet-data export-bundle \
  --catalog "$DATA_ROOT/manifests/potsdam-catalog.json" \
  --split configs/splits/potsdam-legacy-v1.json \
  --preprocessing-id isprs-canonical-v1 \
  --preprocessing-dir "$DATA_ROOT/manifests/preprocessing-v2" \
  --model-input-id isprs-irrg-zscore-v1 \
  --data-root "$DATA_ROOT" \
  --output "$DATA_ROOT/bundles/potsdam-irrg-zscore-v1"

deepnet-data verify-bundle \
  --bundle "$DATA_ROOT/bundles/potsdam-irrg-zscore-v1"
```

The exporter copies the view's raster assets and split labels to a new
destination. Five-channel Potsdam export requires an nDSM alignment record.

When a qualified co-registration check establishes the nDSM transform, put the
evidence and method in a versioned copy of the dataset descriptor. Rebuild the scene
catalog with `build-catalog --dataset-config-dir <descriptor-directory>` and retain
that catalog as a new data version. The resulting evidence is covered by the catalog
content hash; raw metadata and pixels stay untouched.

## Consume the same bundle locally or on Kaggle

Verify the bundle once before training, then use the shared access API. The bundle
holds relative file URIs so the same artifact works when mounted from a local disk
or `/kaggle/input`.

```python
from pathlib import Path

from deepnet.data.bundles import load_bundle
from deepnet.data.dataset import ScenePatchDataset, make_torch_dataloader
from deepnet.runtime import configure_torch_runtime, seed_everything

runtime = configure_torch_runtime()  # CUDA, then MPS, then CPU
seed_everything(42)
bundle = load_bundle(Path("/path/to/verified/bundle"))
train_access = bundle.scene_access("train")
dataset = ScenePatchDataset(
    train_access,
    window_size=(256, 256),
    epoch_size=1000,
    seed=42,
)
loader = make_torch_dataloader(
    dataset,
    batch_size=4,
    **runtime.dataloader_options(),
)
```

The training runners are `deepnet.training` (joint), `deepnet.potsdam_study`
(Potsdam DINOv2) and `deepnet.potsdam_rgbir.study` (Potsdam DINOv3). In a custom
loop, count the labels in sampled, augmented batches before sending them to the
accelerator:

```python
import logging

class_names = {
    0: "impervious surfaces",
    1: "buildings",
    2: "low vegetation",
    3: "trees",
    4: "cars",
    5: "clutter/background",
}
logger = logging.getLogger("deepnet.training")
histogram = dataset.new_label_histogram(class_names=class_names)

for epoch in range(num_epochs):
    dataset.set_epoch(epoch)
    histogram.reset()
    for batch in loader:
        histogram.update(batch["labels"], batch["valid_mask"])
        train_step(batch)
    histogram.log(logger, epoch=epoch + 1)
```

Each JSON log records per-class shares of valid semantic pixels and patches
containing that class. It counts ignored labels and invalid pixels separately.

Pass a model to `TorchSegmentationAdapter` to check input channel count and
layout.

On Kaggle, enable the GPU accelerator, mount the wheel and bundle datasets, and use
the absolute input paths shown in Kaggle's Input panel. Save checkpoints, logs, and
metrics under `/kaggle/working`; persist them through the Kaggle notebook's output
version. On a Mac, use an external data/output directory so Git remains small and
raw inputs remain untouched.

## Reproducibility and validation

The bundle manifest records scene catalog, split, preprocessing, and model-input
hashes plus every copied file's byte count and SHA-256. The run manifest adds seed,
sampler/config hashes, Git revision and dirty state, Python version, and runtime
metadata. Keep the JSON run manifest beside checkpoints and metrics.

Run the synthetic contract tests and source checks locally:

```bash
ruff check \
  deepnet scripts model_script semantic_labeling tests notebooks_src test_kernel/test.py
ruff format --check \
  deepnet scripts model_script semantic_labeling tests notebooks_src test_kernel/test.py
pytest tests/unit
```

The unit suite uses synthetic fixtures. For an offline integration check on a
Potsdam patch with pinned B/L weights, see `python -m
deepnet.potsdam_rgbir.model smoke --help`. CUDA preflight adds small-fit quality
and throughput checks.
