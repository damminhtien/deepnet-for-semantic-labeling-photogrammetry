# Repository architecture

`deepnet.data` defines raster contracts and reads source data. Dataset
descriptors name the inputs; scene catalogs record file metadata and optional
hashes. Split manifests assign scenes before patch sampling. Model-input specs
select named bands and define normalization.

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

Training applies geometric augmentation to aligned bands, labels and masks
before normalization. Optical transforms affect optical bands only. Validation
and test use deterministic windows and the labels declared in their split.
Fitted statistics use training windows and record the split hash.

The DINOv3 RGBIR pipeline caches scene arrays before sampling. Cache manifests
record source hashes, split roles, label type, preprocessing and feature
statistics. The Rasterio reader verifies cache contents and supplies windows to
other runners.

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

Georeferenced rasters must share CRS and affine grid or be reprojected.
Pixel-local rasters require evidence of source alignment, such as the ISPRS
image/label pairs for Vaihingen. Five-channel Potsdam exports require a
co-registration record for the nDSM JPEGs.

`potsdam-legacy-v1` keeps the earlier 18/6/14 scene assignment for comparisons
with prior runs. Neighboring scenes can belong to different splits.

## Local and Kaggle runtimes

Local and Kaggle runs install the same package and verify the same bundle.
`deepnet.runtime` selects CUDA, MPS or CPU in that order. Pass data paths at
runtime. Write local outputs outside the repository and Kaggle outputs to
`/kaggle/working`.

See [the data pipeline guide](data-pipeline.md) for installation, catalog, bundle,
and Kaggle GPU commands.

The model package accepts a feature backbone and adds optical/auxiliary adapters
and a segmentation decoder. `from_pretrained` loads weights on demand. DINOv2
and DINOv3 have separate specs; SAM requires a feature adapter. Model configs
record LoRA targets and license references. `apply_lora` also accepts an
injected encoder.

The historical PSPNet and SAM runners use their model-specific bundles and the
helpers in `semantic_labeling/`. `deepnet/training.py` runs joint DINOv2
training with balanced batches, masked CE/Dice, resumable checkpoints,
full-scene validation and final test evaluation. `deepnet/training_utils.py`
handles learning rates, pretrained weights and artifacts. Kaggle installs the
runner and checks input hashes. The [Potsdam coverage
study](experiments/potsdam-dinov2-coverage-refit.md) uses separate jobs and
bundles for selection, refit and scoring.

The [DINOv3 RGBIR pipeline](experiments/potsdam-dinov3-rgbir-v2.md) sends RGB
bands to the pretrained encoder. Its optional gated detail branch uses IR, the
IR/red index computed after augmentation, and DSM relief and availability. The
feature adapter uses DINOv2 `layernorm` or DINOv3 `norm` and checks CLS/register
token counts. Defaults for normalization and gating preserve older model specs.

## Completed Potsdam DINOv3 path

Selection compares A/B/C on 18 training and six validation scenes, then repeats
the winner from pretrained weights with seed 43. A won with RGB, baseline
augmentation and identity inference. Its ViT-B/16 blocks 3/6/9/12 share a patch
grid. The decoder resizes their features to fuse them with the stride-four
detail branch. See the [final
review](experiments/potsdam-dinov3-rgbir-v2-final-review-2026-11-06.md) for
results.

`deepnet/potsdam_rgbir/protocol.py` records selection rules, seeds, TTA, source
hashes and GPU limits. `phases.py` continues the selected seed-42 checkpoint for
1,152 updates on a separate cache with 24 participant-labeled scenes. It scores
the final checkpoint once on the historical benchmark, using $512 \times 512$ windows
and stride 256.

The joint Potsdam/Vaihingen DINOv3 plan is unexecuted. Its [splits and evaluation
rules](data-pipeline.md#unexecuted-joint-dinov3-split-and-evaluation-protocol)
are separate from the completed Potsdam-only campaign.

## Completed Potsdam PSPNet comparison

[PSPNet RGB](experiments/potsdam-pspnet-rgb-v1.md) uses
[`model_script/pspnet_rgb.py`](../model_script/pspnet_rgb.py) for the ResNet101
encoder and pyramid pooling head. It shares the DINOv3 campaign's role-scoped
caches, sampling and full-scene evaluator. The
[Kaggle coordinator](../scripts/run_pspnet_rgb_kaggle.py) verifies the separate
selection and 24-scene refit artifacts before opening the benchmark cache for
one evaluation. This is the matched RGB comparison, separate from the older
five-channel PSPNet runner and its historical outputs.
