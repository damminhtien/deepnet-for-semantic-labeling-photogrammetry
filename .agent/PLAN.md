# DINOv2 full-run milestones

1. [x] Correct post-radiometry invalid-pixel fill, add a focused regression, and
   replay the train-only rare-class/padding audit before enabling augmentation.
2. [x] Add a pinned DINOv2 joint-run configuration/protocol and reusable runner:
   balanced class-aware sampling, masked CE/Dice, LoRA optimizer, accumulation,
   mixed precision, warmup/cosine, resumable checkpoints, validation-only selection,
   and one post-selection full-scene test evaluation per dataset.
3. [x] Verify synthetic end-to-end training/resume and real pretrained forward/
   backward locally; run relevant checks and verify four immutable data bundles.
4. [x] Package source, pinned weights, and data as private Kaggle inputs; verify
   package hashes and launch one full GPU job with durable outputs and logs.
5. [x] Review attempt 1 failure, preserve evidence, repair the runtime, and
   verify actual GPU optimizer updates in a separate attempt with frozen inputs.
6. [x] Retrieve and audit final checkpoint, validation/test metrics, per-scene
   confusion matrices, provenance, and artifact hashes; assess scientific results
   and commit the final run state.

Record observed results after each milestone. Do not describe static checks,
synthetic tests, or a running job as completed scientific training.

## Potsdam coverage and refit study

1. Verify residual/decoder behavior and real pretrained forward/backward.
2. Validate the DSM source snapshot, feature grid, train-only statistics and
   refit bundle boundaries; preserve previous split artifacts.
3. Run the CUDA small-fit and timing pilot before freezing the selection schedule.
4. Select on 18/6, confirm the winner, continue on 24, then score 14 once.
