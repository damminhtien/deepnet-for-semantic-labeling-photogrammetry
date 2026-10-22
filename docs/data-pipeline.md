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
label counts, deterministic IRRG histograms, and aligned before/after augmentation
examples. It saves tables and figures under `DEEPNET_ARTIFACT_ROOT`, outside Git.

The analysis uses notebook-style cells in
[`notebooks_src/data_eda_augmentation.py`](../notebooks_src/data_eda_augmentation.py).
Install `.[notebooks,vision,data-versioning]`, materialize the DVC inputs, and run
`dvc repro`. Open the source with a Jupytext-enabled editor or convert it to an
`.ipynb` using:

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

### Frozen EDA snapshot

The current analysis was generated against DVC lock SHA-256
`3cff56c46a3471b123e3e7c74c1c09fa25e192d9d4b662693605516e27b9eb0d`. Label
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

The optical profile uses two deterministic 256 x 256 windows from every active scene
in each role. It is a scene-stratified sample, not a full-raster histogram. Values
below are sampled p05 / median / p95 digital numbers, in source band order IR, red,
green.

| Dataset / role | IR | Red | Green |
| --- | --- | --- | --- |
| Potsdam train | 47 / 96 / 163 | 49 / 83 / 150 | 42 / 75 / 150 |
| Potsdam validation | 44 / 97 / 169 | 46 / 87 / 169 | 40 / 78 / 165 |
| Vaihingen train | 41 / 122 / 208 | 37 / 72 / 149 | 36 / 76 / 139 |
| Vaihingen validation | 41 / 106 / 211 | 42 / 71 / 162 | 39 / 73 / 142 |

The alignment report passed all 54 active image/label scene pairs. Potsdam IRRG has
CRS/affine metadata while its label TIFFs are pixel-local and use source-attested
same-scene pairing. Vaihingen inputs and labels are pixel-local and use explicit
ISPRS image/label pair evidence. Matching dimensions alone are never the acceptance
criterion.

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
