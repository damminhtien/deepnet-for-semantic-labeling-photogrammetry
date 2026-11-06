# Active task decisions — 2026-11-05

- Implement the approved `potsdam-dinov3-rgbir-v2` campaign with 24 new allocated
  GPU-hours. Count all assigned GPUs and bootstrap/retry execution. Do not charge
  historical runs against this budget or change gates after looking at results.
- Preserve the historical 18/6/14 scene split. Start selection/confirmation from
  pinned pretrained DINOv3, not the old checkpoint already refit on 24 scenes.
- Use independently verified RGB as encoder input. Record the observed historical
  IRRG-to-RGBIR mapping 4/2/3 without rewriting old inputs or evidence. Compute
  the new IR/red index from RGBIR bands 4/1 after optical augmentation.
- Continue the eligible seed-42 winner only after seed-43 confirmation; reset
  AdamW/scaler, train exactly 1,152 updates on 24 participant-label scenes, then
  score the locked final checkpoint once. The historical benchmark has already
  been observed, and OA >=99% remains a research target.
- Keep all source/data/weight uploads private and all large artifacts outside
  Git. Local static/offline checks do not establish CUDA throughput or accuracy.

## Frozen selection — 2026-11-06

- A wins the eight-pass seed-42 ranking and D passes the same four floors with
  seed 43. Continue A's RGB/baseline-augmentation weights for the exact refit.
- Keep identity inference with 512/256 overlap. Hflip improves OA by only
  0.0785799/0.0689946 point, below the required 0.1 for either seed. Preserve
  these measured negative TTA decisions; do not revise the threshold afterward.
- The canonical selection-lock hash is
  `6408c4737250c4bc0c0d27a28b6e325b4928ed38e7cfc49ec0a203c524d998b8`.
  Cache preparation and upload recovery do not change this lock or the protocol.

- Final benchmark is complete with the update-1,152 refit weights and identity
  policy: OA 92.0397%, mIoU 81.0932%, evaluation count one. Retain the measured
  car-IoU regression and the two scene-OA regressions. OA 99% is unmet; this
  completed campaign does not revise its frozen selection using benchmark labels.

## Completed preceding campaign decisions — 2026-10-31

- The user's request to run DINOv2 supersedes the completed audit task's temporary
  prohibition on training. Colab remains canceled; use the existing Kaggle path.
- Use DINOv2 ViT-B/14, LoRA rank 8, joint three-channel IRRG training, and the existing
  scene-level Potsdam 18/6/14 and Vaihingen 12/4/17 splits. DINOv3 configs stay frozen.
- Use 504-pixel Potsdam and 280-pixel Vaihingen windows, divisible by patch stride
  14. Balance dataset microbatches equally and use train-only class-aware sampling.
- Select the checkpoint by the unweighted mean of the two separate validation
  mIoUs. Freeze its hash/configuration before opening either test bundle; persist
  each test's per-scene and aggregate counts. Retain historical-use qualifications.
- Pin pretrained revision and runtime dependencies. Protect Kaggle's installed
  accelerator-specific Torch build. Keep data/source/weight uploads private and
  output artifacts local-only; preserve original raw and historical artifacts.
