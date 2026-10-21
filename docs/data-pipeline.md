# Data pipeline guide

This document describes the common path from external raster sources to a
reproducible local or Kaggle run. Large data and outputs belong outside Git.

## Install

The package requires Python 3.10 or newer. For local development, create an
isolated environment and install the data, training, and lint/test dependencies:

```bash
uv sync --extra dev --extra train
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
  --preprocessing-id isprs-canonical \
  --data-root "$DATA_ROOT" \
  --tile-size 1024 \
  --view-assets irrg-view=irrg \
  --output "$DATA_ROOT/manifests/potsdam-quality-v1.json"
```

Each required view is checked against the per-scene reference grid and the label
asset selected by the split. Unknown valid label colors, missing assets, bad hashes,
and unsupported alignment fail explicitly.

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
  --preprocessing-id isprs-canonical \
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
  --preprocessing-id isprs-canonical \
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
