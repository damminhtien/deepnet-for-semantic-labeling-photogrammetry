# Completed follow-up: multimodal segmentation architecture

## Status

Architecture, versioned configs, synthetic coverage, and documentation are complete.
No pretrained weights were downloaded and no training job was started. The five-band
Potsdam plan remains blocked on verified nDSM decoding and spatial registration.

## Goal

Implement a configurable multi-domain segmentation model on top of the shared data
pipeline. Use a replaceable pretrained ViT backbone, domain-aware spectral and
auxiliary-input adapters, a DPT-style multi-scale decoder with a high-resolution
skip, and deterministic Albumentations policies. Keep DINOv3 as the research
configuration, DINOv2 as an alternative, and SAM as an injectable baseline. Do not
start model training in this task.

## Acceptance

- Domain and augmentation specs are versioned, typed, and validated independently
  of model architecture.
- Training augmentation operates on raw aligned bands before model-view
  normalization; geometry synchronizes all bands, labels, and valid masks, while
  radiometric transforms touch configured optical bands only.
- RGB inputs retain an identity path into the pretrained normalization; IRRG uses a
  learned spectral projection, and elevation inputs use a separate spatial branch.
- Backbone loading is lazy and configurable. Tests use a small fake backbone and do
  not download pretrained weights.
- The multi-scale decoder accepts selected intermediate features and a high-resolution
  spatial feature map, returns the requested class logits, preserves original image
  size after patch-stride padding, and propagates gradients.
- Model and training JSON configs encode the selected Potsdam defaults and preserve
  the DINOv3 license limitation and DINOv2 alternative.
- Focused tests cover augmentation synchronization, optical-only photometric
  changes, spectral channel selection, shape handling, deterministic seeds, and
  forward/backward gradients.

## Scope

Implement reusable modules, configuration, tests, and documentation only. Do not
download model weights, run a Kaggle job, or train a checkpoint. Potsdam elevation
training remains unavailable until the nDSM alignment evidence is verified.

# Completed foundation: contract-driven data pipeline

## Goal

Build a scalable, contract-driven data layer that future model callers can use on a local Mac or Kaggle GPU through thin, configurable adapters. Old experiment outputs are historical and must be rerun under this pipeline before they are treated as current.

# Acceptance

- Dataset IDs resolve through a catalog to immutable, per-scene asset records with checksums and verified raster metadata.
- `RasterAsset`, `SceneRecord`, `WindowSpec`, `SplitManifest`, `PreprocessingSpec`, `ModelInputSpec`, and `Sample` contracts enforce their invariants.
- Potsdam's existing scene split is represented as a versioned manifest. The shared data API takes split IDs from manifests rather than embedding them in loaders.
- CRS, affine transform, resolution, band semantics, nodata, and explicit alignment evidence are checked. Equal shapes alone never establish spatial alignment.
- Canonical preprocessing is deterministic and versioned. Any fitted statistics use training scenes only. Model views select channels and normalization without changing raw data.
- Shared lazy window reading, patch/grid sampling, augmentation, scene evaluation, and portable bundle export are available independently of model architecture.
- A thin Torch adapter validates the tensor contract without selecting or constructing a model.
- Local MPS/CPU and Kaggle CUDA runtimes use the same package and bundle contracts; the Kaggle wheel install leaves its preinstalled Torch untouched.
- Run manifests record dataset, split, preprocessing, model-view, sampler, source revision, and artifact provenance.
- Old experiment outputs and raw data remain untouched and outside Git; notebooks and docs mark old metrics for rerun.
- Focused tests cover contract validation, split leakage, alignment, labels, deterministic sampling, coverage, bundle hashes, and provenance.

# Scope

Implement the common data layer, its local/Kaggle packaging path, and initial catalog support for available ISPRS Potsdam and Vaihingen sources. Keep prior model runners and notebook outputs explicitly historical; do not make this data-layer task depend on old experiment code or train models. Model-specific callers can be added when the experiments are rerun.
