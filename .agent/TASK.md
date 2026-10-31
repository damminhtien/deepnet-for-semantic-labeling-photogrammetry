# Train DINOv2 through the complete shared pipeline

The user authorized an end-to-end DINOv2 training run on 2026-10-31. Complete the
shared training/evaluation caller, remove the known augmentation-padding blocker,
validate it locally, and launch one full private Kaggle GPU job using DINOv2
ViT-B/14 with LoRA and the existing joint Potsdam/Vaihingen IRRG data.

Preserve the scene splits, raw rasters, historical experiments, DINOv3 configs, and
unrelated edits. Keep test bundles separate until validation selects and freezes
the checkpoint; score each test set once and retain per-scene confusion matrices.
Persist resumable checkpoints and runtime/data/model/configuration provenance.
Keep outputs outside Git and retrieve verified run artifacts locally. No Colab
restart, public data publication, Git push, or DVC push is requested.
