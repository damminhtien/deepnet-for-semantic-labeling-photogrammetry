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
