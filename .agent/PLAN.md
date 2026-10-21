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
