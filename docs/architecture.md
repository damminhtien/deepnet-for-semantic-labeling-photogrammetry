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
                             verified role-specific memmap cache
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

The DINOv3 RGBIR campaign materializes verified scene arrays between source
reading and sampling. Cache manifests bind source bytes, split roles, label type,
preprocessing and feature statistics. The shared Rasterio reader remains the
reference for cache equivalence; other runners can sample directly from it.

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
| `configs/training/` | Versioned joint-training plans and Potsdam study settings |
| `deepnet/data/contracts.py` | Immutable versioned data contracts and canonical hashes |
| `deepnet/data/catalog.py` | Catalog discovery, source inspection, and schema loading |
| `deepnet/data/augmentations.py` | Lazy Albumentations construction with synchronized band and mask transforms |
| `deepnet/data/raster.py` | CRS/affine/dtype/nodata verification and lazy Rasterio reads |
| `deepnet/data/preprocessing.py` | Label decoding, model views, and train-only statistics |
| `deepnet/data/sampling.py` | Random, class-aware, epoch histogram, deterministic grid, and coverage sampling |
| `deepnet/data/dataset.py` | Split-bound scene access, patches, histogram setup, collation, DataLoader creation |
| `deepnet/data/evaluation.py` | Full-scene, overlap-weighted logits and confusion metrics |
| `deepnet/data/bundles.py` | Content-addressed portable model-view bundles |
| `deepnet/data/lineage.py` | Run manifest builder with data/source/runtime provenance |
| `deepnet/runtime.py` | CUDA/MPS/CPU detection and execution defaults |
| `deepnet/potsdam_study.py` | Potsdam pilot, validation selection, train-only refit, and held-out evaluation |
| `deepnet/potsdam_rgbir/` | Verified RGBIR caches, deterministic sampling, DINOv3 pilot, selection/TTA, fixed refit and historical benchmark |
| `deepnet/models/specs.py` | Versioned backbone, input adapter, and task-head contracts |
| `deepnet/models/backbones.py` | Hugging Face feature extraction and injected backbone interface |
| `deepnet/models/adapters.py` | Optical projection and separate auxiliary spatial features |
| `deepnet/models/decoders.py` | DPT-style fusion of four block features and a high-resolution detail branch |
| `deepnet/models/segmenter.py` | Composition that pads inputs and returns dense task logits |
| `deepnet/models/adapter.py` | Thin input/output contract around a caller-provided Torch model |
| `model_script/` | Model definitions, thin shared-data bundle commands, and historical training runners |
| `scripts/run_potsdam_rgbir_kaggle.py` | Private phase packaging, submission receipts, gated continuation and downloaded artifact audits |
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
points. `deepnet/training.py` now orchestrates balanced joint DINOv2 training,
masked CE/Dice, resumable optimizer/RNG checkpoints, full-scene validation, and
post-selection test evaluation. `deepnet/training_utils.py` holds persistence,
learning-rate, pretrained verification, and artifact inventory helpers. The
Kaggle bootstrap installs the packaged runner and verifies private input hashes.
The [Potsdam coverage study](experiments/potsdam-dinov2-coverage-refit.md) uses
separate jobs and bundles for validation selection, refit and held-out evaluation.
Its verified DSM relief and availability mask are separate from the older
five-channel view, which requires verified nDSM decoding and co-registration.
The notebooks mark their saved outputs as historical pending reruns.

The [DINOv3 RGBIR campaign](experiments/potsdam-dinov3-rgbir-v2.md) uses a separate
config and package. Named R/G/B bands reach the pretrained encoder; IR, the
post-augmentation IR/red index and DSM relief/mask enter a gated detail branch.
The feature adapter resolves DINOv2 `layernorm` and DINOv3 `norm` and validates
CLS/register token counts. New normalization/gating fields default to the old
behavior, preserving existing serialized model contracts. The campaign requires
real pretrained gradient checks in preflight and does not derive quality claims
from synthetic unit tests.

## Completed Potsdam DINOv3 path

The [final artifact review](experiments/potsdam-dinov3-rgbir-v2-final-review-2026-11-06.md)
records the completed campaign. Selection compares A/B/C on 18 training and six
validation scenes, then confirms the winner from pretrained with seed 43. The
winning A recipe uses RGB, baseline augmentation and identity inference. Its
ViT-B/16 features at blocks 3/6/9/12 share one patch grid; the decoder resizes
them for fusion with its stride-four detail branch.

`deepnet/potsdam_rgbir/protocol.py` binds selection eligibility, both seeds, TTA,
source hashes and allocation limits. `phases.py` continues the locked seed-42
checkpoint for exactly 1,152 updates on a separate 24-scene participant-label
cache. Former validation scenes become training data at that boundary and cannot
select another checkpoint. Only the completed final checkpoint can enter the
single historical benchmark evaluation, using 512x512 windows and stride 256.

The joint Potsdam/Vaihingen DINOv3 configuration remains an unexecuted plan.
Its split and ranking settings do not control the completed Potsdam-only
campaign. See the [data pipeline guide](data-pipeline.md) for the two scopes.

## Completed Potsdam PSPNet comparison

[PSPNet RGB](experiments/potsdam-pspnet-rgb-v1.md) uses
[`model_script/pspnet_rgb.py`](../model_script/pspnet_rgb.py) for the ResNet101
encoder and pyramid pooling head. It shares the DINOv3 campaign's role-scoped
caches, sampling and full-scene evaluator. The
[Kaggle coordinator](../scripts/run_pspnet_rgb_kaggle.py) verifies the separate
selection and 24-scene refit artifacts before opening the benchmark cache for
one evaluation. This is the matched RGB comparison, separate from the older
five-channel PSPNet runner and its historical outputs.
