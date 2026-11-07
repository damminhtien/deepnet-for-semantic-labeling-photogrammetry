# Potsdam PSPNet RGB comparison on Kaggle

Implement and run `potsdam-pspnet-rgb-v1` using the winning DINOv3 data and
training protocol as a fixed comparison. Keep the 18/6/14 roles, use verified
RGB and baseline augmentation, refit the selected PSPNet weights on 24
participant-label scenes, then score the same historical benchmark once.
Use private Kaggle jobs. The canceled PSPNet Colab campaign and the completed
DINOv3 benchmark remain unchanged. Keep model weights, data and logs outside Git.

# DINOv3 documentation and default-branch integration

Completed on 2026-11-06: documentation and all 19 campaign-branch commits were
fast-forwarded into the existing default `master`. Origin was verified at
`42c77367ff956e35416fcb5c005cb0b9dadcbf44`. All 85 unit tests and documentation
checks pass. The unrelated PSPNet edit was restored unchanged and is excluded
from integration commits. The following task-state commit records this closure.

The user requested documentation updates and a merge into the main branch on
2026-11-06. The repository's existing default branch is `master`; there is no
`main` branch locally or on origin. Deliver the completed campaign on that
existing default branch without renaming it.

Synchronize the README, architecture, data guide and invariants with the audited
Potsdam DINOv3 result. Distinguish the completed Potsdam-only campaign from the
unexecuted joint DINOv3 plan. Validate documentation against saved artifacts and
run the unit/source checks before merging. Preserve the unrelated PSPNet state
edit, historical evidence and large external artifacts. This request authorizes
the Git integration; no DVC push, input publication or new GPU run is requested.

## Completed Potsdam DINOv3 RGBIR study

Implement the user-approved `potsdam-dinov3-rgbir-v2` study (2026-11-05): real
RGB to DINOv3-B/16, optional satellite L/16 pilot, aligned IR/NDVI/DSM auxiliary
features, verified memory-mapped caches, deterministic sampling/augmentation,
and a new 24 allocated-GPU-hour Kaggle budget. Preserve 18/6/14 selection roles.
After seed-43 confirmation, continue seed 42 on all 24 participant-label scenes
for exactly 1,152 updates and score the locked final model once. OA >=99% is a
research target; report measured outcomes and preserve negative results.

Completed on 2026-11-06: final update-1,152 checkpoint scored once on the
historical 14-scene benchmark. Independent OA is 92.0397%, six-class mIoU 81.0932%
on 467,862,855 pixels; total allocation is 13.4557472 GPU-hours. OA 99% remains
unmet by 6.9603 points. See the final 2026-11-06 DINOv3 artifact review.

Keep historical artifacts immutable. Use the previous C validation confusion
matrix as the four-metric eligibility reference. Historical test results have
already been observed; do not describe the 14-scene benchmark as fresh holdout.
Keep data/checkpoints/logs outside Git. Commit focused changes, preserve unrelated
edits. The campaign's original no-push instruction is superseded for Git by the
documentation/integration request above; DVC and input publication stay excluded.

## Completed preceding task

### Train DINOv2 through the complete shared pipeline

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
