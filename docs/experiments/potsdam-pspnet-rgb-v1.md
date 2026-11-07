# Potsdam PSPNet RGB comparison

This campaign compares PSPNet–ResNet101 with the winning DINOv3-B RGB recipe
reported in the [DINOv3 final review](potsdam-dinov3-rgbir-v2-final-review-2026-11-06.md).
The DINOv3 reference is 92.0397% OA and 81.0932% six-class mIoU. The comparison
uses the same historical benchmark cohort and does not isolate backbone effects:
the segmentation heads, pretrained data, and trainable parameter counts differ.

PSPNet starts from [TorchVision ResNet101 ImageNet-1K V2 weights](https://docs.pytorch.org/vision/2.0/models/generated/torchvision.models.resnet101.html#torchvision.models.ResNet101_Weights).
The SHA-256 is `cd907fc2a0de2292b2ba2b27d2216bc672d84632d1cf20058d01c0c03b5e020e`.
It uses RGB normalization, stride-16 context, pyramid pooling at 1/2/3/6 bins,
GroupNorm in the decoder and a stride-four detail path. The encoder's BatchNorm
running statistics stay frozen with microbatch one. All backbone parameters and
the new head are trainable. The old five-channel PSPNet checkpoint is preserved
as a separate experiment.

| Phase | Data, schedule and gate |
| --- | --- |
| Selection | Reuse the DINOv3 cache, training index, sampler, class weights, baseline augmentation, CE/Dice and seed 42. Fit eight patches to at least 90% mIoU within 512 updates before discarding fit weights. Profile on Kaggle and proceed only if the matched schedule fits with 25% headroom. Run eight passes of 864 updates; select among full validations at passes 4 and 8 by OA, then mIoU, then earlier step. |
| Refit | Load the selected PSPNet checkpoint and reset AdamW/scaler. Continue on the exact 24-scene participant-label cache for 1,152 updates. Encoder/head LR are 1e-4/3e-4 in selection and half those values in refit. |
| Benchmark | Only the audited refit at update 1,152 opens the separate test cache. Score once with identity inference, 512 window, stride 256, six classes and 467,862,855 eroded-reference pixels. |

This is a fixed-recipe, single-seed comparison. PSPNet can qualify for refit even
if its validation score is lower than DINOv3. The 17 allocated GPU-hour worker
ceilings are 12 selection (including pilot), 3 refit and 2 benchmark. The
coordinator records provider execution time for both allocated T4s. If the pilot
cannot fit the matched schedule, stop and report its measured timing.

Run from the repository environment:

```bash
python -m scripts.run_pspnet_rgb_kaggle campaign --run "$RUN" --reference "$DINO_RUN"
```

Use a new external run directory and the audited DINOv3 campaign as the
reference. Existing private Kaggle datasets are attached by phase; hashes and
label roles are checked inside each job. Preserve accepted submission receipts;
resume the coordinator from those receipts after a monitoring interruption.
An unconfirmed submission needs reconciliation before another push. Failed
phases keep their output and do not open later caches. The final comparison
is written to `comparison.json` after confusion matrix and artifact audits.
