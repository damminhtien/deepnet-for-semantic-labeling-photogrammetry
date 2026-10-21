# Goal

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
