# Decisions

## 2026-10-21: Adopt a contract-driven data layer

- Put the reusable pipeline in a small `deepnet.data` package alongside the existing model code. Use frozen dataclasses and versioned JSON manifests to keep the core schema dependency-light and portable.
- Use Rasterio for geospatial inspection, reprojection, and lazy window access. A matching array shape is insufficient evidence of spatial alignment; assets without georeferencing require explicit source/coregistration evidence or remain unusable for aligned views.
- Keep raw source assets immutable and outside Git. Catalogs, split manifests, preprocessing specs, bundle manifests, and quality reports contain metadata and relative URIs only.
- Freeze the current Potsdam 18/6/14 scene assignment as `potsdam-legacy-v1`. The split enforces scene/group disjointness but is not represented as a region-held-out split; new research claims require new evaluation.
- Treat previous experiment metrics and checkpoints as historical. The new pipeline does not promise checkpoint or score compatibility; rerun evaluation before using old numbers as current evidence.
- Keep prior model training loops as historical records for this data-layer change. New model callers supply their architecture through a thin adapter when experiments are rerun; the pipeline itself does not construct PSPNet, U-Net, SAM, or another model.
- Package the shared API as a pure-Python wheel for Kaggle. Install it without dependency resolution so Kaggle's accelerator-specific Torch build remains intact; verify Rasterio and other runtime dependencies separately.
- Use one host-aware runtime configuration on local Mac and Kaggle: CUDA first, Apple MPS second, CPU fallback. Real Kaggle CUDA execution remains a later experiment, while the local MPS path and mocked CUDA selection are validated here.

## 2026-10-19: Publish aggregate evidence only

Publish numeric aggregate metrics, split/protocol descriptions, code/checkpoint hashes, and dataset source citations. Keep raw imagery, labels, archives, trained weights, feature caches, predictions, previews, and logs outside the public repository.

## 2026-10-19: Keep experiment scores separate from official leaderboard values

The local probes use frozen encoders and coarse scoring grids. Report their exact split and resolution and do not compare them as if they were full-resolution ISPRS submissions. The official Potsdam table identifies the participant as `SWJ`; no `SWJ_2` row was found during this review.

## 2026-10-19: Treat TerraMind as provisional, not final

TerraMind leads the sampled RGB validation probes and Potsdam held-out OA; UniverSat leads the deeper Vaihingen mIoU and is marginally ahead on Potsdam held-out mIoU. Recommend TerraMind for the next prototype and retain UniverSat as challenger, while noting that production-sized latency and memory were not measured.

## 2026-10-19: Keep SAM and sidewalk claims bounded by evidence

The SAM ViT-B LoRA implementation and private cloud job were prepared, but completed training metrics were not verified. Tile2Net is the strongest direct sidewalk candidate found in the research; its published paper result must not be attributed to the current beta model.

## 2026-10-19: Push the full local master history

The user explicitly authorized pushing all commits on local `master`, including the six earlier local commits. Keep the publication as a new forward commit and verify the remote branch after push.

## 2026-10-20: Persist cloud reruns outside ephemeral runtimes

Write Colab run artifacts into a unique subfolder of the existing project Drive folder while training proceeds; keep the latest resumable checkpoint and hash-verified logs/manifests there and copy them to local project Data. Kaggle SAM remains a separate job and consumes the fixed Potsdam dataset plus a small private source-code dataset so its imports are present in `/kaggle/input` without re-uploading the multi-gigabyte image dataset. Keep Kaggle outputs local only; do not upload them to Drive.
