# Contract-driven data pipeline migration

## M1: Contracts, catalog, and frozen splits

- Add typed immutable contracts, canonical JSON serialization/hashes, dataset ID catalog, source inventory tooling, and the `potsdam-legacy-v1` split manifest.
- Make the 18/6/14 membership and scene-level disjointness explicit; report the legacy split's spatial scope honestly.

Validation: unit tests for schema parsing, required fields, canonical hashes, duplicate scenes/groups, unknown scenes, and split leakage.

## M2: Raster validation and preprocessing

- Add Rasterio-backed asset inspection and lazy window reads with CRS/affine/resolution/band/nodata checks.
- Add explicit alignment policies, label encoding, quality reports, deterministic preprocessing specs, train-only statistics, and generic named model-input views.

Validation: synthetic GeoTIFF fixtures exercise geospatial alignment, resampling, nodata, label palette errors, and train-only statistics. Inspect the real Potsdam/Vaihingen headers without modifying raw sources.

## M3: Shared data access and evaluation

- Implement deterministic scene/window/patch datasets, uniform and class-aware samplers, synchronized stochastic augmentation, grid coverage, and full-scene confusion aggregation.
- Provide shared split-bound datasets, DataLoader settings, evaluation, and a thin Torch adapter with no architecture-specific construction.
- Keep old training/evaluation code as historical reference; mark notebook outputs and published metrics stale until rerun through the shared API.

Validation: focused deterministic sampling, augmentation alignment, complete coverage, and adapter tensor-contract tests; inspect each edited notebook source through Jupyter MCP.

## M4: Portable bundles and lineage

- Route data preparation through one exporter driven by dataset, split, preprocessing, and model-view IDs.
- Use the same source scene catalog for configurable channel views; include exact manifests and verify bundle hashes before a consumer reads them.
- Record pipeline IDs and hashes in run manifests and evaluation outputs.

Validation: round-trip small synthetic bundles, path traversal rejection, checksum mismatch detection, and cloud package inclusion checks.

## M5: Documentation and completion

- Document setup, catalog authoring, data-root configuration, immutable artifacts, split limitations, model views, and migration commands.
- Document local Mac and Kaggle GPU setup, runtime preflights, and the boundary between the new data API and historical trainers.
- Refresh the repository Graphify index and review the exact diff.

Validation: run focused tests and static checks requested by repository instructions, `git diff --check`, and a Graphify update. Do not claim model training or historical metrics were reproduced.

## Follow-up: configurable multimodal segmentation architecture

Status: implementation complete. Training and pretrained-weight download were
intentionally not performed.

### M6: Domain and augmentation contracts

- Add typed domain/band metadata and versioned augmentation transform specs.
- Add a domain-specific Albumentations factory and integrate it into training samples
  before deterministic normalization.
- Keep samplers free of augmentation logic and seed transforms per sample.

Validation: synthetic multi-band samples confirm synchronized image/label/valid
geometry, optical-only radiometric changes, and repeatability by sample seed. Passed
with the pinned Albumentations 2.0.8 API and configured Potsdam policy.

### M7: Replaceable backbones and spectral adapters

- Add lazy Hugging Face ViT loading for configurable DINOv3 and DINOv2 checkpoints.
- Add an injectable feature-backbone interface for SAM and future encoders.
- Add an identity RGB path, learned IRRG projection, and a separate auxiliary-band
  feature branch.

Validation: fake backbones verify feature selection, prefix-token removal, arbitrary
patch-stride padding, and channel semantics without downloading weights. Passed for
RGB identity mapping, learnable IRRG projection, and separate auxiliary inputs.

### M8: Multi-scale decoder and model configuration

- Add DPT-style feature reassembly/fusion with a high-resolution spatial skip.
- Compose the input adapter, backbone, and decoder into one segmentation model.
- Add versioned model and Potsdam training configs, including optimizer, loss,
  schedule, sampling, stopping, and checkpoint-selection settings.

Validation: CPU forward/backward tests cover three- and five-channel views, output
shape, finite gradients, and loss ignore-index behavior. Passed for patch strides 14
and 16; local MPS synthetic forward/backward also passed.

### M9: Documentation, quality checks, and handoff

- Document model choices, license restrictions, optional dependencies, local/Kaggle
  setup, and remaining Potsdam alignment evidence.
- Run targeted unit tests, Ruff, formatting, `git diff --check`, and Graphify update.
- Commit the implementation in focused commits. Do not launch training.

Validation: 32 unit tests, Ruff, formatter, diff check, local MPS smoke test, pure
Python wheel build/install, packaged config loading, and Graphify update passed. No
remote Kaggle execution was started. The DINOv3 Hub weights require user access and
license acceptance before explicit loading.

## Follow-up: DINOv3 training preparation for Potsdam and Vaihingen

### M10: DVC data versioning and portable handoff

Status: implementation and local DVC handoff complete (2026-10-21). DVC catalogs,
quality reports, split-specific bundles, and remote objects were verified for both
datasets. The staged DVC snapshot excludes Vaihingen's 17 post-selection labels.
No Kaggle session was used for validation or piloting.

- Initialize DVC without adding raw rasters or credentials to Git. Keep machine-local
  cache and remote locations in ignored local configuration.
- Define versioned data inputs for both datasets and document how to materialize the
  exact DVC revision locally or package it once for Kaggle.
- Stage only the labels assigned to train and validation; keep Vaihingen's 17
  post-selection references outside this DVC snapshot.
- Reuse existing Kaggle data when the scene inventory and hashes match; do not start
  a GPU session for validation or piloting.

Validation: DVC metadata parses, source manifests resolve, and no raster/checkpoint
bytes or machine credentials are staged in Git.

### M11: Dataset splits, DINOv3 views, and robust augmentation

Status: the DINOv3 split and evaluation protocol v2 are locked (2026-10-22); the
full-scene EDA snapshot below is historical DINOv3 v1 evidence. The 19-cell Jupyter
notebook ran all 10 code cells after a project `.venv` kernel restart; its Jupytext
source also completed as a fresh process. Both runs used the materialized DVC
snapshot at lock SHA `17994ddb59a804bc`. Every IRRG pixel in 38 Potsdam scenes
(18 train and 20 v1 validation, including the 14 challenge-test tiles) and 16
Vaihingen v1 train/validation scenes was scanned. All 38 Potsdam DSM/nDSM scene
pairs were inspected and hashed; 37 DSM grids match IRRG metadata, scene `3_13` is
one column short, and nDSM registration remains unverified for every scene. The
audit did not read Vaihingen test pixels or labels. It also replayed the v2
synchronized augmentation policy over three seeds per training scene. Results and
provenance are under
`/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/eda/` outside Git.

- Keep the benchmark-aligned Potsdam v2 split: 18 train, 6 internal validation,
  and 14 challenge-heldout test scenes. The 18/6 division is project-defined within
  ISPRS's 24 labeled tiles; hold test labels out of training and selection bundles.
- Keep test-only DVC exports in `pipelines/evaluation/dvc.yaml`, outside the default
  train/validation graph. The single-run Kaggle caller must evaluate each test role
  once after freezing the selected checkpoint, and save per-tile/aggregate
  confusion matrices plus the ISPRS table-style class F1 and overall accuracy.
- Use 12 participant-labeled Vaihingen scenes for training, 4 eroded-reference
  scenes for validation, and keep 17 official references for post-selection
  evaluation. Keep unverified Potsdam nDSM out of the unified run.
- Expand the config-driven aerial augmentation policy with synchronized geometry,
  ignored affine borders, and optical-only radiometric transforms; review it against
  source histograms and paired image/label examples in a reproducible EDA notebook.

Validation: split and training-config hashes are checked against the locked
evaluation protocol; data-quality reports cover all active train/validation scenes;
full-scene IRRG histograms are exact per scene; the elevation audit records grid,
coverage, and encoding limits. The DINOv3 v2 Potsdam test split is excluded from the
train/validation bundle and selection. Its local labels were used in historical v1
EDA/experiments; v2 scores them only after checkpoint freeze. A complete
training/evaluation caller is still required. No training or v2 test evaluation was
run.

### M12: One unified Kaggle run plan and README

Status: configuration and documentation complete; training is deferred. DINOv3
weights remain access-gated, and the user has requested approval. No GPU pilot or
training session was started. The plan requires one full Kaggle run after access,
weight, bundle, and durable-output preflights pass.

- Configure one multi-domain DINOv3 run with balanced dataset sampling, separate
  validation metrics, deterministic manifests, and resumable checkpoints. Potsdam
  selection uses its 6-scene internal validation set; its 14 challenge-heldout
  scenes and Vaihingen's 17 official test scenes remain outside selection.
- Document local Mac preparation, DVC materialization, Kaggle packaging/secrets, and
  the single-run policy in the README and data-pipeline guide.
- Download weights only after Hugging Face grants access; do not start Kaggle before
  all data, package, license, checkpoint, and durable-output preflights pass.

Validation: local static and synthetic checks only; no Kaggle GPU pilot. Commit and
push focused milestones after reviewing the exact diff.

### M13: Rare-class, boundary, and padding augmentation audit

Status: audit complete (2026-10-22); the augmentation and training configs remain
unchanged pending review. A train-only scan covered 12,121 overlapping candidate
windows across 30 scenes. Five focused crops per dataset were replayed with eight
deterministic seeds each. It found that two of 16 Potsdam car-containing trials
removed the rare class from a low-density crop, and optical transforms changed
affine-padding values despite correct ignore labels and validity masks. Model
patch-stride padding was zero for both configured crop sizes. Tables, provenance,
and visual examples are stored outside Git in the EDA artifact directory.

- Review per-class retention and semantic-edge density for car and clutter/background.
- Fix the photometric-padding interaction, or make optical transforms mask-aware,
  before full training; rerun the focused audit after any policy change.
- Review class-aware sampling or a bounded geometric policy for sparse rare-class
  crops. Keep the validation and official test distributions untouched.

Validation: the project `.venv` kernel was restarted, then the notebook's analysis
cells ran through Jupyter MCP. The final artifact-export cell source ran in that same
fresh kernel through the Jupyter MCP code tool because the notebook document endpoint
reported zero cells for that final index. Ruff and Python compilation passed; no
training was run.
