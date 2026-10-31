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
5. [ ] Verify live training progress, persist exact run identity/state, retrieve
   and audit final artifacts when available, and commit the implementation/state.

Record observed results after each milestone. Do not describe static checks,
synthetic tests, or a running job as completed scientific training.
