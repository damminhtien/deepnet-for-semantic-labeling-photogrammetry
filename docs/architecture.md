# Repository architecture

`deepnet.data` is the source of truth for raster contracts and data access. Dataset
descriptors are keyed by stable IDs; per-scene catalogs capture observed file
metadata and optional content hashes. Split manifests assign complete scenes before
patch sampling. A model-input spec maps named source bands and normalization rules to
the tensor expected by a model.

## Data flow

```text
raw rasters (read-only)
        │
        ▼
dataset descriptor ── build-catalog ──► scene catalog + source hashes
        │                                      │
        └──────── split manifest ──────────────┤
                                               ▼
                                    alignment and quality checks
                                               │
                                               ▼
                            lazy window reader + canonical labels/masks
                                               │
                             split-aware sampling and augmentation
                                               │
                                               ▼
                               model view / DataLoader
                                               │
                                               ▼
                  spectral adapter + auxiliary spatial feature branch
                                               │
                                               ▼
                      replaceable backbone + multi-scale decoder
                                               │
                                               ▼
                       chunked scene evaluation + run provenance
```

The raw tree is never modified by catalog, validation, training, or bundle export.
Canonical preprocessing is deterministic and versioned. Domain-specific geometric
augmentation is applied to raw aligned bands, labels, and validity masks before
model-view normalization; optical radiometry is applied to optical bands only.
Validation and test use deterministic windows and the split's declared labels.
Fitted channel statistics are computed from train windows and bind to the exact
split hash.

## Core modules

| Path | Role |
| --- | --- |
| `configs/datasets/` | Dataset descriptors for ISPRS Potsdam and Vaihingen |
| `configs/splits/` | Frozen scene-level split manifests |
| `configs/preprocessing/` | Label ontology and deterministic raster policies |
| `configs/model-inputs/` | Declarative named-band selections and normalization |
| `configs/domains/` | Versioned sensor and band-role descriptions |
| `configs/augmentations/` | Domain-scoped stochastic training transforms |
| `configs/models/` | Backbone, spectral adapter, decoder, task, and license settings |
| `configs/training/` | A configured Potsdam run plan; it has not been executed |
| `deepnet/data/contracts.py` | Immutable versioned data contracts and canonical hashes |
| `deepnet/data/catalog.py` | Catalog discovery, source inspection, and schema loading |
| `deepnet/data/augmentations.py` | Lazy Albumentations construction with synchronized band and mask transforms |
| `deepnet/data/raster.py` | CRS/affine/dtype/nodata verification and lazy Rasterio reads |
| `deepnet/data/preprocessing.py` | Label decoding, model views, and train-only statistics |
| `deepnet/data/sampling.py` | Random, class-aware, deterministic grid, and coverage sampling |
| `deepnet/data/dataset.py` | Split-bound scene access, patches, collation, DataLoader creation |
| `deepnet/data/evaluation.py` | Full-scene, overlap-weighted logits and confusion metrics |
| `deepnet/data/bundles.py` | Content-addressed portable model-view bundles |
| `deepnet/data/lineage.py` | Run manifest builder with data/source/runtime provenance |
| `deepnet/runtime.py` | CUDA/MPS/CPU detection and execution defaults |
| `deepnet/models/specs.py` | Versioned backbone, input adapter, and task-head contracts |
| `deepnet/models/backbones.py` | Hugging Face feature extraction and injected backbone interface |
| `deepnet/models/adapters.py` | Optical projection and separate auxiliary spatial features |
| `deepnet/models/decoders.py` | DPT-style four-level fusion and high-resolution skip |
| `deepnet/models/segmenter.py` | Composition that pads inputs and returns dense task logits |
| `deepnet/models/adapter.py` | Thin input/output contract around a caller-provided Torch model |
| `model_script/` | Model definitions, thin shared-data bundle commands, and historical training runners |
| `model_kernel/` | Historical notebooks awaiting reruns against the new pipeline |

## Alignment and dataset scope

Shape equality never proves that two geographic rasters are aligned. Georeferenced
rasters must share CRS and affine grid or be explicitly reprojected. Pixel-local
rasters require source-pair alignment evidence. Potsdam nDSM is a JPEG without
geospatial metadata or a verified co-registration record, so current five-channel
exports are blocked. Vaihingen sources that have no CRS/affine are represented as
pixel-local, with evidence limited to the official scene-matched image/label pair.

`potsdam-legacy-v1` preserves the earlier 18/6/14 scene assignment. Its scope is
scene-level, not region-held-out. It is a rerun protocol for comparison with older
work, not a newly independent test cohort.

## Local and Kaggle runtimes

Both environments install the same package and verify the same bundle manifest.
`deepnet.runtime` selects CUDA on Kaggle when available, MPS on a supported Mac, or
CPU otherwise. Data paths are CLI/runtime arguments rather than model constants.
Bundles are immutable inputs; local artifacts are written outside the repository,
and Kaggle outputs go to `/kaggle/working` for explicit artifact persistence.

See [the data pipeline guide](data-pipeline.md) for installation, catalog, bundle,
and Kaggle GPU commands.

The shared model package provides an injectable feature-backbone interface, a lazy
Hugging Face wrapper, spectral and spatial adapters, and a multi-scale segmentation
head. Pretrained weights load only when a caller explicitly requests
`from_pretrained`. DINOv3 and DINOv2 have separate model specs; SAM can be supplied
through the same interface with an external feature adapter. LoRA target names are
configuration data and `apply_lora` can wrap a caller-supplied encoder. License
references are recorded with each model config.

The previous PSPNet and SAM training loops and the old modules under
`semantic_labeling/` are retained as historical experiment code. They still consume
the previous model-specific bundle format and are not the current training entry
points. The new architecture does not add a training runner; sampling orchestration,
loss execution, checkpointing, and evaluation remain caller-owned. The configured
Potsdam five-channel plan requires verified nDSM decoding and coregistration before
data access. The notebooks mark their saved outputs as historical pending reruns.
