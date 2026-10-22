# Data pipeline guide

This document describes the common path from external raster sources to a
reproducible local or Kaggle run. Large data and outputs belong outside Git.

## Install

The package requires Python 3.10 or newer. For local development, create an
isolated environment and install the data, training, model, augmentation, and
lint/test dependencies:

```bash
uv sync --extra dev --extra train --extra models --extra vision --extra data-versioning --extra notebooks
```

On Apple Silicon, the runtime selects MPS when the installed Torch build exposes
it. Run this short preflight after installation to record the actual interpreter,
Torch version, and selected device:

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

The wheel contains the command-line entry point and default JSON contracts. Check
that `numpy`, `rasterio`, and the existing `torch` import before running. If Rasterio
is missing, install a compatible Rasterio wheel in that notebook environment; do
not reinstall Torch as part of data setup.

For DINOv2/DINOv3 or domain-specific augmentation, install the optional packages in
the active Kaggle environment while retaining its existing Torch package:

```python
%pip install "albumentations==2.0.8" "transformers>=4.57,<6" "peft>=0.14,<1"
```

The local and Kaggle installs read the same packaged JSON configuration. Pretrained
weights are fetched only when the user code explicitly invokes the configured
backbone loader.

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

Choose a root containing the source folders declared by a dataset config. The
descriptor paths are relative to this root and do not include machine-specific
prefixes.

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

Catalog creation reads source headers and can stream a SHA-256 per asset once. It
does not normalize, rewrite, or copy a raw raster. A catalog is a snapshot: rebuild
it as a new version when source inventory or metadata changes.

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
asset selected by the split. Unknown valid label colors outside the configured
palette tolerance, missing assets, bad hashes, and unsupported alignment fail
explicitly.

## Explore the data and review augmentation

Set `DEEPNET_DATA_ROOT` to the materialized input directory. The analysis reads the
versioned catalogs and full-scene quality reports without modifying source rasters,
then evaluates only the train and validation roles in the selected split. It reports
per-scene asset coverage and raster metadata, explicit alignment evidence, full-scene
label counts and IRRG histograms, auxiliary elevation checks, and aligned
before/after augmentation examples. It saves tables and figures under
`DEEPNET_ARTIFACT_ROOT`, outside Git.

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
cells from the top; result tables, figures, and provenance are saved outside Git
under `DEEPNET_ARTIFACT_ROOT`. To regenerate the notebook from the companion source,
run:

```bash
uv run jupytext --to ipynb \
  --output notebooks/data_eda_augmentation.ipynb \
  notebooks_src/data_eda_augmentation.py
```

Set `DEEPNET_DATA_ROOT` to `data/raw/inputs` and `DEEPNET_ARTIFACT_ROOT` to a
directory outside the repository before running. The analysis marks reference-grid
checks separately from metadata: matching width/height by itself is never treated as
proof of alignment.
Potsdam uses eroded labels for all 20 validation scenes; Vaihingen uses eroded labels
for its 4 validation scenes. Training reads participant labels. The 17 Vaihingen
post-selection scenes are not scanned for label scores during model selection.

The current canonical preprocessing v2 retains the official ISPRS RGB palette and
allows a maximum 80-point Euclidean RGB distance for source color-managed labels.
This source-specific need was confirmed by comparing Potsdam scene 4_12's full
noBoundary label to its exact-palette participant noBoundary counterpart: every
non-ignore pixel decoded to the same class, with a maximum nearest-palette distance
of 69.88. Tied colors or colors farther than the configured tolerance still fail.
The official six-class palette and ignored black boundaries are documented by
[ISPRS](https://isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx).

### DINOv3 split and evaluation lock

The joint DINOv3 run uses the versioned [evaluation protocol](../configs/evaluations/isprs-dinov3-v1.json)
and its two [Potsdam](../configs/splits/potsdam-dinov3-v1.json) and
[Vaihingen](../configs/splits/vaihingen-dinov3-v1.json) split manifests. The protocol
binds each manifest by ID, version, and canonical content hash. It also binds the
training plan to SHA-256
`b5e170e4f1192b587a6d97f8db798c5fcbc0bc6d07edadd4c4a006f394df7124`; the EDA
notebook refuses to run if that plan changes without updating the protocol.

Exact scene membership is:

- **Potsdam train (18):** `2_10`, `2_11`, `2_12`, `3_10`, `3_11`, `5_10`, `5_12`,
  `6_10`, `6_11`, `6_7`, `6_8`, `6_9`, `7_10`, `7_11`, `7_12`, `7_7`, `7_8`, `7_9`.
- **Potsdam validation (20):** `2_13`, `2_14`, `3_12`, `3_13`, `3_14`, `4_10`,
  `4_11`, `4_12`, `4_13`, `4_14`, `4_15`, `5_11`, `5_13`, `5_14`, `5_15`,
  `6_12`, `6_13`, `6_14`, `6_15`, `7_13`. Potsdam has no test scenes in this run.
- **Vaihingen train (12):** `3`, `5`, `13`, `15`, `17`, `21`, `23`, `26`, `30`,
  `32`, `34`, `37`.
- **Vaihingen validation (4):** `1`, `7`, `11`, `28`.
- **Vaihingen post-selection test (17):** `2`, `4`, `6`, `8`, `10`, `12`, `14`,
  `16`, `20`, `22`, `24`, `27`, `29`, `31`, `33`, `35`, `38`.

Checkpoint selection uses the unweighted mean of the Potsdam and Vaihingen
validation mIoU values. Report each dataset's metrics and confusion matrix
separately; sum confusion matrices only within a dataset, and never pool pixels
across datasets. Validation uses full-scene deterministic grids with overlap-weighted
logit aggregation. Ignore ID 6 and invalid pixels are excluded. Fitted statistics
use training scenes only, and validation/test augmentation is disabled.

The 17 Vaihingen test scenes remain unopened during training and model selection.
Run their eroded-label evaluation once after the final training configuration and
selected checkpoint are frozen and recorded in the run manifest. This measures
post-selection cross-dataset transfer for this run; related scenes appeared in
other research, so describe the result accordingly. The split is scene-disjoint, not
region-disjoint, and neighboring tiles may share geographic context. Do not claim
region-independent generalization from this protocol. The current joint DINOv3 plan
uses the three IRRG bands. The separate
[`potsdam-dinov3-multimodal-v1` plan](../configs/training/potsdam-dinov3-multimodal-v1.json)
uses the older `potsdam-legacy-v1` split and is not covered by this joint-run
protocol; it remains gated on the DSM/nDSM audit below.

### Frozen EDA snapshot

The current analysis was generated against DVC lock SHA-256
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

The car class is about 1% of semantic pixels in both datasets. Vaihingen clutter is
more concentrated: it appears in 4/12 training scenes and 1/4 validation scenes.
This supports the configured 30% class-aware component for training windows; it
does not justify changing the natural validation distribution.

The 2026-10-22 optical profile scans every train and validation IRRG pixel in native
raster blocks. It covers all 38 Potsdam scenes (18 train, 20 validation) and the 16
Vaihingen scenes in this split (12 train, 4 validation); no Vaihingen test raster or
label was read. Histograms and quantiles are exact for uint8 values. Every scanned
IRRG pixel passed its raster mask and finite-value checks.

The table reports pooled band means and medians plus the range of per-scene medians
in source band order IR, red, green. The per-scene CSV retains exact minima,
percentiles, standard deviations, valid-pixel counts, and full 256-bin histograms.

| Dataset / role | Scenes / pixels per band | Mean IR / red / green | Pooled median IR / red / green | Per-scene median range IR / red / green |
| --- | ---: | --- | --- | --- |
| Potsdam train | 18 / 648,000,000 | 98.71 / 92.57 / 85.55 | 95 / 85 / 77 | 79–119 / 78–91 / 71–86 |
| Potsdam validation | 20 / 720,000,000 | 96.68 / 92.52 / 86.25 | 92 / 86 / 79 | 80–105 / 78–94 / 71–88 |
| Vaihingen train | 12 / 58,554,459 | 121.39 / 82.01 / 80.86 | 121 / 72 / 76 | 98–150 / 58–101 / 61–97 |
| Vaihingen validation | 4 / 19,533,347 | 114.82 / 79.61 / 78.70 | 112 / 71 / 74 | 101–118 / 64–73 / 70–76 |

Vaihingen's IR channel is brighter and more variable than Potsdam's in this split;
Vaihingen's red channel is lower. The per-scene ranges also show substantial
within-dataset variation, especially in Vaihingen training scenes. Keep these
differences visible in view-specific normalization and diagnostics; do not derive
training statistics from validation pixels. The full outputs include
`irrg-band-statistics.csv`, `irrg-per-scene-band-statistics.csv`, and
`irrg-full-histograms.csv` under the external EDA artifact directory.

The alignment report passed all 54 active image/label scene pairs. Potsdam IRRG has
CRS/affine metadata while its label TIFFs are pixel-local and use source-attested
same-scene pairing. Vaihingen inputs and labels are pixel-local and use explicit
ISPRS image/label pair evidence. Matching dimensions alone are never the acceptance
criterion.

### Potsdam DSM/nDSM alignment and value audit

The notebook audited all 38 Potsdam train/validation DSM and configured lastools
nDSM files from `DEEPNET_AUX_DATA_ROOT`; it hashed each file and scanned its values
by native blocks. These 76 elevation files are not in the DVC input snapshot (0/76
found under `data/raw/inputs`), so their hashes and audit outputs are provenance
records, not DVC-managed source versions.

| Asset | Result against the matching IRRG reference grid | Scene match vs pixel-grid evidence |
| --- | --- | --- |
| DSM GeoTIFF | 37/38 have matching EPSG:32633 CRS, affine transform, width, and height (6000 x 6000). | Scene `3_13` has the same CRS and affine origin/resolution but is only 5999 x 6000; one 6000-pixel column is missing. It is not a complete grid match. |
| nDSM JPEG | 37/38 have the same 6000 x 6000 dimensions; `3_13` is 5999 x 6000. | All files lack CRS/affine metadata and the catalog has no co-registration evidence. Matching the scene name and shape does not prove pixel alignment. |

The DSM TIFFs have no embedded vertical-unit metadata; the dataset config describes
them as metres, but the source unit and vertical datum still need authoritative
confirmation. Full scans found DSM values from -35.75 in train scene `7_10` to
106.171 in validation scene `7_13`. These extrema need source review before choosing
elevation clipping or normalization. DSM per-scene mean/min/max and standard
deviation use all valid pixels; its p01–p99 values use a deterministic,
block-stratified sample of about 100,000 pixels per scene. The nDSM JPEG stores
uint8 digital numbers from 0 to 255; its histograms and quantiles are exact encoded
value summaries, not metric heights.

Do not train the five-channel view from these files yet. First add immutable DSM and
nDSM sources to DVC with checksums and source provenance; restore the full `3_13`
rasters or explicitly record a supported common-coverage mask; obtain authoritative
nDSM decoding and co-registration evidence; then validate all grids and fit elevation
statistics on training scenes only. The joint DINOv3 configuration currently uses
IRRG only, so it does not consume the unverified elevation channels.

### Augmentation policy and audit

`isprs-irrg-aerial-v2` uses all square symmetries and a 35% affine transform with
scale 0.9–1.1, rotation ±12 degrees, translation ±4%, linear image interpolation,
nearest-neighbor labels, and ignore-filled borders. Brightness/contrast, gamma,
Gaussian noise, and 3–5 pixel blur are independently sampled with probabilities
35%, 20%, 15%, and 10%. Photometric transforms operate only on IRRG and leave
labels, validity masks, and auxiliary bands unchanged. The IRRG values are
multispectral, so an RGB hue/saturation transform is intentionally excluded.

The augmenter derives a separate deterministic random stream for each transform.
This matters because the pinned Albumentations `Compose.set_random_seed()` assigns
the same seed to every child transform; without separate child seeds, their
probability gates are correlated. A regression test verifies that two 50% optical
transforms can appear in all four apply/skip combinations while replaying the same
sample seed exactly.

The audit samples one training crop from each train scene and replays it with three
seeds (54 Potsdam and 36 Vaihingen samples). Every source class present in these
sampled windows remained represented after geometry. The fifth-percentile valid
area was 88.9% for Potsdam and 88.8% for Vaihingen; median valid area was 100% in
both. Conditional on a photometric change, the median absolute change was 5.9–6.8
DN per Potsdam band and 3.8–4.3 DN per Vaihingen band. These are policy diagnostics,
not performance results or a guarantee for every crop. Affine-created borders stay
ignored, and validation is never augmented.

Review `augmentation-examples.png`, `augmentation-impact.png`,
`augmentation-audit.csv`, and `augmentation-summary.csv` before changing the
versioned policy. `eda-provenance.json` records the source hash, DVC lock, catalogs,
splits, quality reports, runtime versions, and seeds.

#### Rare classes, semantic boundaries, and padding follow-up

The expanded audit is restricted to the DINOv3 training scenes (18 Potsdam and 12
Vaihingen); validation and official test scenes were not used for crop selection or
augmentation replay. It scanned 12,121 overlapping candidate windows at half-window
stride (9,522 Potsdam and 2,599 Vaihingen). A class is treated as rare when its
training-pixel share is at most 6% or it appears in at most half the training scenes.
This selects car and clutter/background in both datasets. Clutter/background is
present in only 4 of 12 Vaihingen training scenes.

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

The two Potsdam car disappearances came from the low-density crop in train scene
`6_7` at `(row=4608, col=3584)`, which contained 1,274 car pixels before
augmentation. Both occurred in separate affine replays. This does not establish a
general failure rate for all Potsdam cars, but it shows that the current geometry can
remove a rare target from a sparse crop. The 4-neighbor measure is a semantic class
boundary, not an instance boundary; these labels do not identify touching objects
of the same class.

| Dataset | Trials with affine padding | Mean / p95 / max padded area | Padding labels ignore and mask invalid | Nonzero padded values, after geometry → after radiometry | Trials where radiometry changed padding | Mean absolute change per padded pixel | Model patch padding |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Potsdam | 45% | 2.95% / 9.89% / 13.50% | 100% / 100% | 2.93% → 27.08% | 30% | 2.84 DN | 0% |
| Vaihingen | 40% | 3.02% / 13.06% / 14.56% | 100% / 100% | 8.23% → 33.33% | 25% | 4.33 DN | 0% |

Padding is labeled and masked correctly, but photometric transforms change the image
values in those invalid pixels. The nonzero percentages and mean absolute changes
are averaged over trials with affine padding; the changed-trial percentage uses all
replay trials. Model patch padding is zero because both crop sizes (512 and 288) are
divisible by DINOv3 ViT-B/16's patch size of 16.

Before full training, restore the configured image fill value after optical
radiometry (or keep radiometric operations mask-aware), then rerun this audit. Also
review a rare-class sampling or bounded geometry policy for low-density car crops.
Neither the versioned augmentation config nor the training config was changed during
this audit, and these diagnostics are not model-performance evidence.

The expanded artifacts are stored outside Git under
`/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/eda/20261022T063946Z/`:
`augmentation-training-window-scan.csv`, `augmentation-rare-boundary-manifest.csv`,
`augmentation-rare-boundary-audit.csv`, `augmentation-rare-boundary-summary.csv`,
`augmentation-padding-audit.csv`, `augmentation-padding-summary.csv`, and
`augmentation-rare-boundary-examples.png`. In the figure, cyan marks semantic class
edges and magenta marks affine-generated padding. `eda-provenance.json` includes the
split/catalog/config hashes, seeds, and runtime details.

The inspected Vaihingen inventory contains 33 image scenes; 17 are missing the
participant-label asset. Define a supervised split from scenes with the chosen
label asset, or use the image-only catalog for inference. Missing labels remain
explicit in the catalog.

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

The exporter includes only bands needed by the view and the split's required
labels. Existing bundle destinations are never overwritten. The current Potsdam
nDSM JPEG has no verified co-registration metadata; a five-channel export therefore
stops at the alignment check until a verified alignment record is supplied. Do not
make it pass by copying the IRRG dimensions into its metadata.

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

Supply a model constructed by the caller to `TorchSegmentationAdapter`. It checks
that model inputs use the configured channel count and layout but does not select
or construct a network. Dataset code never imports a model architecture.

On Kaggle, enable the GPU accelerator, mount the wheel and bundle datasets, and use
the absolute input paths shown in Kaggle's Input panel. Save checkpoints, logs, and
metrics under `/kaggle/working`; persist them through the Kaggle notebook's output
version. On a Mac, use an external data/output directory so Git remains small and
raw inputs remain untouched.

The local preflight selects MPS on a supported Apple Silicon setup and CUDA on a
GPU runtime such as Kaggle. The CPU fallback is intended for data preparation and
small smoke runs. This verifies device discovery only; model training must still
be launched by the caller's training loop.

## Reproducibility and validation

The bundle manifest records scene catalog, split, preprocessing, and model-input
hashes plus every copied file's byte count and SHA-256. The run manifest adds seed,
sampler/config hashes, Git revision and dirty state, Python version, and runtime
metadata. Keep the JSON run manifest beside checkpoints and metrics.

Run the synthetic contract tests and source checks locally:

```bash
ruff check deepnet tests/unit
ruff format --check deepnet tests/unit
pytest tests/unit
```

Tests use small synthetic GeoTIFFs, not the private source data or cloud runtimes.
They verify data contracts; they do not establish model quality or GPU training
completion.
