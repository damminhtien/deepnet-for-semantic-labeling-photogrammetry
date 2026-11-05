# Potsdam DINOv3 RGBIR v2 milestones

1. [x] Implement named RGB/auxiliary routing, DINOv3 normalization, gated fusion,
   compact normalized decoder and pinned model specs; verify real pretrained LoRA.
2. [x] Verify RGBIR provenance, create scoped caches, implement deterministic
   pixel-centered rare sampling and mask-aware baseline/enhanced augmentation.
3. [x] Implement config-driven selection/confirmation/TTA/refit/benchmark phases,
   immutable bindings, budget pilot, resume and final-only scoring guards.
4. [x] Run focused regression checks, real-data cache/model checks, and update
   docs/Graphify; commit changes by subsystem.
5. [ ] Package private Kaggle inputs and run the CUDA preflight within two GPU-hours;
   freeze model roster and the common 12/8/6-pass schedule only if gates pass.
6. [ ] Run eligible selection arms and seed 43 within 16 GPU-hours; lock TTA/recipe.
7. [ ] Refit 24 scenes for 1,152 updates, score once if eligible, audit artifacts,
   report measured results and allocated GPU-hours, and commit final task state.

## Completed DINOv2 full-run milestones

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

1. [x] Verify residual/decoder behavior and real pretrained forward/backward.
2. [x] Validate the DSM source snapshot, feature grid, train-only statistics and
   refit bundle boundaries; preserve previous split artifacts.
3. [x] Run the CUDA small-fit and timing pilot before freezing the selection schedule.
4. [x] Select on 18/6 and verify the seed-43 confirmation against all four floors.
5. [x] Continue the locked seed-42 checkpoint on 24 scenes; verify the final
   update-1,152 weights, bundle, optimizer state and artifact inventory.
6. [x] Score the 14 held-out scenes once and independently recompute all metrics
   on 467,862,855 valid pixels; verify the final checkpoint and attempt binding.
7. [x] Record final results, budget and limitations in the README and artifact
   review, validate the documentation and commit the task-owned changes.
