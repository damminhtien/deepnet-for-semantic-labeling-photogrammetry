# Active task decisions — 2026-10-31

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
