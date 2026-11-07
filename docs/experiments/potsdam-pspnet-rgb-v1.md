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

## Verified results, 2026-11-07

The [selection job](https://www.kaggle.com/code/damminhtien/potsdam-pspnet-rgb-v1-selection-run)
passed the CUDA small fit at update 64 (94.2980% train mIoU). On the six validation
scenes, PSPNet improved from 91.2254% OA / 77.6931% mIoU at update 3,456 to
92.5307% OA / 79.0996% mIoU at update 6,912. The latter checkpoint was selected.

The [refit job](https://www.kaggle.com/code/damminhtien/potsdam-pspnet-rgb-v1-refit-run)
loaded that checkpoint and completed exactly **1,152 updates** on 24 train scenes,
with 1,165,743,019 valid pixel presentations. The final SHA-256 is
`a1a1a82842714262300ddccff909ed5001205124150e139665d45c8c7eed6986`.
The [benchmark job](https://www.kaggle.com/code/damminhtien/potsdam-pspnet-rgb-v1-benchmark-run)
version 1 scored that checkpoint once. Both models used the same historical
14-scene cache, eroded labels, 512×512 windows, stride 256, and identity inference.
The audit recomputed OA, IoU and F1 from the aggregate and per-scene matrices and
confirmed identical class support on **467,862,855 scored pixels**.

| Benchmark metric | PSPNet RGB refit | DINOv3-B RGB refit | PSPNet minus DINOv3 |
| --- | ---: | ---: | ---: |
| OA | 91.2834% | 92.0397% | -0.7563 pp |
| Six-class mIoU | 79.4235% | 81.0932% | -1.6697 pp |
| Incorrect pixels | 40,781,843 | 37,243,220 | +3,538,623 |
| Full-scene scoring time | 254.83 s | 405.23 s | 37.1% less time |
| Allocated GPU-hours, matched phases | 1.9066 | 3.8218 | 50.1% less |

GPU-hours sum selection, refit and benchmark for PSPNet and DINOv3 arm A. This
excludes the other DINOv3 arms, confirmation and separate preflight. Scoring time
covers full-scene scoring rather than model forward alone.

| Class | PSPNet IoU | DINOv3 IoU | IoU difference (pp) | PSPNet F1 | DINOv3 F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Impervious surfaces | 87.9163% | 88.7372% | -0.8209 | 93.5697% | 94.0325% |
| Building | 94.1508% | 95.0554% | -0.9047 | 96.9873% | 97.4650% |
| Low vegetation | 78.1639% | 80.0577% | -1.8938 | 87.7438% | 88.9245% |
| Tree | 80.7474% | 81.8287% | -1.0813 | 89.3484% | 90.0064% |
| Car | 92.2606% | 92.4418% | -0.1812 | 95.9745% | 96.0725% |
| Clutter/background | 43.3019% | 48.4383% | -5.1364 | 60.4345% | 65.2638% |

PSPNet trails in OA on 13 of 14 scenes and has lower IoU in all six classes.
The largest class gap is clutter, where PSPNet recall is
53.85% versus 60.39% for DINOv3.
The measured tradeoff is lower scoring time and GPU cost with lower accuracy for
the tested PSPNet recipe. This single-seed comparison of complete systems does
not isolate the encoder or establish the same tradeoff on a new dataset.

<details>
<summary>OA by benchmark scene</summary>

| Scene | PSPNet OA | DINOv3 OA | PSPNet minus DINOv3 (pp) |
| --- | ---: | ---: | ---: |
| 2_13 | 91.3616% | 92.7240% | -1.3624 |
| 2_14 | 92.3210% | 93.3365% | -1.0155 |
| 3_13 | 92.0192% | 92.6437% | -0.6245 |
| 3_14 | 90.5486% | 91.7142% | -1.1656 |
| 4_13 | 90.6583% | 90.9456% | -0.2873 |
| 4_14 | 85.0377% | 85.9410% | -0.9033 |
| 4_15 | 92.1651% | 93.4261% | -1.2611 |
| 5_13 | 91.9218% | 92.1974% | -0.2756 |
| 5_14 | 89.8406% | 89.7140% | +0.1265 |
| 5_15 | 90.8708% | 91.5754% | -0.7045 |
| 6_13 | 94.2666% | 94.7947% | -0.5281 |
| 6_14 | 93.8286% | 94.7824% | -0.9538 |
| 6_15 | 92.1693% | 93.3710% | -1.2017 |
| 7_13 | 90.8642% | 91.3106% | -0.4463 |

</details>

The run receipts, hashes, and matrices remain under the external run directory
`runs/potsdam-pspnet-rgb-v1`. The aggregate comparison is in `comparison.json`.
