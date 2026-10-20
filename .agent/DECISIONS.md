# Decisions

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
