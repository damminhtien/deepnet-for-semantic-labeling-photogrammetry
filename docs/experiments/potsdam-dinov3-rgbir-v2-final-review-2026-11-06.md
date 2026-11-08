# Potsdam DINOv3 RGBIR v2 final artifact review, 2026-11-06

**Status: CURRENT** — Canonical final DINOv3 campaign record.

DINOv3-B RGB reached **92.0397% OA and 81.0932% six-class mIoU** on
**467,862,855 valid pixels** from 14 historical benchmark scenes. The benchmark
scored the final 1,152-update refit checkpoint once. Relative to DINOv2, OA rose
**0.5612 percentage points** and mIoU rose **1.4810 points**. The campaign
used **13.4557472 of 24 allocated GPU-hours**, including bootstrap, the failed
preflight and both T4s.

## Final benchmark results

All rows use the same 14 scene IDs, scored counts and ground-truth class
supports. The audit verified the historical metrics-file hashes. Byte equality
of the underlying historical reference maps remains unverified.

| Experiment | OA | Six-class mIoU |
| --- | ---: | ---: |
| SAM | 87.8142% | 73.7024% |
| DINOv2 refit | 91.4785% | 79.6122% |
| DINOv3 RGB refit | 92.0397% | 81.0932% |

![Class IoU and per-scene OA differences](figures/potsdam-dinov3-rgbir-v2-final-2026-11-06.png)

The figure compares aggregate class IoUs and per-scene OA differences using
audited tables derived from saved confusion matrices. The
[SciencePlots rendering](../figures.md#dinov3-reports) reproduces it. DINOv3 uses
identity inference with $512 \times 512$ windows and stride 256; DINOv2 retains 504/252,
and SAM uses its historical tiling. Scores compare complete recipes.

| Class | DINOv2 refit IoU | DINOv3 RGB refit IoU | Difference | DINOv3 F1 |
| --- | ---: | ---: | ---: | ---: |
| Impervious surfaces | 87.8091% | 88.7372% | +0.9281 pp | 94.0325% |
| Building | 94.6559% | 95.0554% | +0.3995 pp | 97.4650% |
| Low vegetation | 79.0312% | 80.0577% | +1.0266 pp | 88.9245% |
| Tree | 81.0786% | 81.8287% | +0.7501 pp | 90.0064% |
| Car | 92.7234% | 92.4418% | -0.2816 pp | 96.0725% |
| Clutter/background | 42.3747% | 48.4383% | +6.0635 pp | 65.2638% |

IoU improves on five of six classes. Clutter gains 6.0635 points, while car IoU
falls by 0.2816 point. Clutter recall rises from 51.3326% to 60.3914%; precision
changes from 70.8307% to 70.9916%. Car recall rises to 98.4353%, but precision
falls to 93.8203%, so its aggregate IoU declines.

The final checkpoint makes **37,243,220 errors**, or **2,625,662 fewer** than
the DINOv2 refit, a **6.59%** reduction.
Tree/low-vegetation confusion in both directions contributes 11,312,236 pixels,
or 30.37% of all errors. The largest directed
confusions are:

| True class | Predicted class | Pixels | Share of all errors |
| --- | --- | ---: | ---: |
| Low vegetation | Tree | 5,975,707 | 16.05% |
| Tree | Low vegetation | 5,336,529 | 14.33% |
| Clutter/background | Impervious surfaces | 3,628,388 | 9.74% |
| Impervious surfaces | Low vegetation | 3,380,493 | 9.08% |
| Low vegetation | Impervious surfaces | 2,510,302 | 6.74% |
| Clutter/background | Low vegetation | 2,261,680 | 6.07% |

## Per-scene results

OA rises over DINOv2 on **12 of 14 scenes**. Aggregate scores use summed
confusion matrices. OA falls on `5_13` (-0.1080 point) and `5_14` (-0.0695
point). Scene `4_14` remains lowest at 85.9410% OA, up 1.3585 points.

| Scene | Scored pixels | DINOv2 OA | DINOv3 OA | OA difference | DINOv3 mIoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| `2_13` | 33,112,031 | 92.2660% | 92.7240% | +0.4579 pp | 80.0330% |
| `2_14` | 33,840,315 | 91.7304% | 93.3365% | +1.6061 pp | 79.4943% |
| `3_13` | 32,965,944 | 92.4143% | 92.6437% | +0.2295 pp | 79.6974% |
| `3_14` | 33,189,588 | 91.5149% | 91.7142% | +0.1993 pp | 80.3907% |
| `4_13` | 33,072,636 | 90.8186% | 90.9456% | +0.1270 pp | 75.6824% |
| `4_14` | 33,228,405 | 84.5825% | 85.9410% | +1.3585 pp | 75.6118% |
| `4_15` | 33,125,809 | 92.7802% | 93.4261% | +0.6460 pp | 78.7794% |
| `5_13` | 33,484,544 | 92.3054% | 92.1974% | -0.1080 pp | 78.4836% |
| `5_14` | 33,332,046 | 89.7836% | 89.7140% | -0.0695 pp | 78.3203% |
| `5_15` | 33,302,999 | 90.3011% | 91.5754% | +1.2742 pp | 80.9597% |
| `6_13` | 33,954,901 | 93.8505% | 94.7947% | +0.9443 pp | 84.1208% |
| `6_14` | 33,566,681 | 94.1487% | 94.7824% | +0.6337 pp | 83.7827% |
| `6_15` | 33,343,530 | 93.3275% | 93.3710% | +0.0436 pp | 84.6902% |
| `7_13` | 34,343,426 | 90.8202% | 91.3106% | +0.4904 pp | 78.4851% |

## Selection, modality and inference decisions

The [selection and confirmation review](potsdam-dinov3-rgbir-v2-selection-summary-2026-11-06.md)
retains all A/B/C/D matrices and the frozen ranking. Selection used 18 original
training and six validation scenes, eight passes and 6,912 updates per arm.
Only full validations at passes 4 and 8 could select a checkpoint, on exactly
199,409,714 eroded-label pixels. Every arm starts from pinned pretrained weights.

A won at **93.0441% validation OA / 80.7498% mIoU**. B scored 92.9923% /
80.6097% with the combined IR/index/DSM branch. C scored 92.9690% / 80.5115%
with stronger augmentation. The final recipe uses RGB and baseline augmentation.

D confirmed A from pretrained weights with seed 43 at **92.9477% / 80.4596%**,
above all four historical C floors. Horizontal flips gained 0.0786 and 0.0690 OA
points for seeds 42 and 43, below the 0.1-point threshold. The selection lock
kept identity inference. Satellite ViT-L passed the
[preflight](potsdam-dinov3-rgbir-v2-preflight-2026-11-05.md) fit check but
exceeded its cost limit.

The source audit matched RGB to the official RGB archive. Historical Potsdam
three-band files match RGBIR 4/2/3 (IR/G/B); their configured IR/red/green names
imply 4/1/2. The new encoder uses 1/2/3 and the auxiliary index uses IR/red 4/1.

## Refit and checkpoint audit

The selection lock records A's seed-42 checkpoint, recipe, fitted statistics,
identity inference and continuation schedule. Refit uses a separate cache with
participant labels for all 24 scenes in the 18+6 union. It resets AdamW and AMP,
uses learning rates $5 \times 10^{-5}$ and $1.5 \times 10^{-4}$, and keeps 5% warmup with cosine decay. Its
3,456 base windows plus 576 draws per rare class give 1,152 updates at
accumulation 4. The checkpoint at the last update goes to the benchmark.

Selection and refit presented 8,159,728,531 valid pixels, 1.78× the
4,575,287,363 presentations in the DINOv2 winning path. These totals count
repeated and augmented draws.

- `final.pt` and `last.pt` have identical model tensors, exact update 1,152 and one
  completed pass. Every model tensor is finite. All five refit inventory files
  match their recorded bytes and SHA-256 values.
- All 211 non-LoRA backbone tensors remain exactly equal to A's selected weights.
  All 96 LoRA tensors, 14 adapter tensors and 80 decoder tensors change.
- All 190 AdamW tensor states are at step 1,152, with the reset initial learning
  rates and expected final cosine rates. The resumed checkpoint contains the
  AMP scaler state.
- Refit records **1,165,743,019 valid pixel presentations**,
  760.5469 optimizer-loop seconds and zero validation seconds.
  Repeated rare-class and augmented samples contribute to presentations; they do
  not count distinct source pixels.
- Refit recorded 0.2609 training loss, 90.7385% train OA and 81.2941% train mIoU
  across augmented training batches.

The initial 36-patch diagnostics exactly match A's selected checkpoint. The final
panel has 9,068,544 valid samples per channel, with zero nonfinite values before
and after normalization. Projected RGB means are 0.3353/0.3459/0.3191. The red
channel has nine values beyond the identity bounds (fraction $9.9244 \times 10^{-7}$), while
green and blue have zero; this small learned residual is preserved in the audit.

| Normalized channel | Mean | Std | Min | Max | p01 / p05 / p50 / p95 / p99 |
| --- | ---: | ---: | ---: | ---: | --- |
| Red | -0.6535 | 0.6004 | -2.0073 | 2.2982 | -1.5208 / -1.3464 / -0.7922 / 0.6743 / 1.2294 |
| Green | -0.4916 | 0.5696 | -1.9357 | 2.3707 | -1.3442 / -1.1698 / -0.6160 / 0.6527 / 1.3994 |
| Blue | -0.3863 | 0.5819 | -1.7686 | 2.5972 | -1.2303 / -1.0575 / -0.5375 / 0.7254 / 1.7489 |

The manifests record input statistics at initialization, small fit, scheduled
validation and refit. The final checkpoint SHA-256 is
`5db9ab78976ad57c2d5d66d974b9ee501abf365c3a685901c726a3cf96b3a420`.

## Single evaluation and cache audit

The single evaluation records the final checkpoint hash, selection lock and
identity policy. The 14 per-scene matrices sum to the aggregate. The review
recomputed OA, mIoU, class IoU, F1, precision and recall for all compared runs.
Inference used 576 overlapping windows per scene, 8,064 total.

The benchmark cache has 73 inventoried files and
5,544,040,723 bytes; all hashes were verified. Its scene IDs, eroded-label role,
class supports and exact scored-pixel count agree with the frozen source bundle.
The independent review compares image/labels/masks exactly on
42 source panels. Scene `3_13` retains zero relief and
availability in its missing DSM column, without reducing scoring support.

End-to-end scoring took **405.2340 seconds**, including reads, transfers,
forward passes, overlap accumulation and reduction. Forward passes took 68.68%
and memmap accumulation took 12.82%. The `raster_io_and_preprocessing` timer
measures reads from the memmap cache here:

| Profile stage | Seconds |
| --- | ---: |
| `memmap_setup` | 5.6646 |
| `raster_io_and_preprocessing` | 16.9420 |
| `host_to_device` | 10.8148 |
| `forward` | 278.3311 |
| `device_to_host` | 12.3925 |
| `memmap_accumulation` | 51.9562 |
| `metric_reduction` | 27.8119 |

## Allocation accounting and recovery

| Phase / attempt | Full execution GPU-hours |
| --- | ---: |
| preflight-failed-v1 | 0.0150355 |
| preflight | 0.3775072 |
| A | 2.9335358 |
| B | 3.2024464 |
| C | 2.8305651 |
| D | 3.2084227 |
| refit | 0.5955538 |
| heldout | 0.2926808 |
| **Total** | **13.4557472** |

The table counts both T4s from execution logs, including logged bootstrap and
wrapper steps. D's total includes TTA.

Probes, full-scene validation, input diagnostics and checkpoint writes took
11.19% / 9.25% / 8.30% / 9.80% of timed training plus validation for A/B/C/D.
The older joint run spent 92.83% on validation under a different schedule and
timing scope.

The coordinator recovered an interrupted upload, waited for dataset processing
and resolved a notebook/dataset name conflict before submission. A status
timeout stopped the local coordinator; retrieval resumed from accepted refit
version 1. The benchmark source path was corrected before cache creation. The
receipts record each recovery.

## Evidence and limits

Campaign root outside Git:
`/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/runs/potsdam-dinov3-rgbir-v2`.

- `study/selection-lock.json`, `study/refit/`, `study/heldout/`: primary bindings,
  training state, final weights, one-time attempt and saved metrics.
- `kaggle/refit/audit.json`, `kaggle/heldout/audit.json`: verified remote inventories,
  source identity and allocation scopes; each phase has an accepted receipt.
- `review-final-20261106/refit-audit.json`, `benchmark-source-audit.json` and
  `benchmark-cache-audit.json`: independent checkpoint, bundle and cache checks.
- `review-final-20261106/analysis.json`, `per-class.csv`, `per-scene.csv`,
  `confusion.csv`, comparison PNG/PDF and `analysis-inventory.json`: derived
  arithmetic and figures. `independent-review.py` and the saved refit/cache review
  scripts preserve the analysis code. The report PNG was visually inspected and
  copied into Git with identical bytes; large artifacts remain external.

The frozen protocol hash is `db17701748df5596704e4da7f2b41d9f5c4d9cab0666e2475e81bfcdbd9edc7d`;
the scientific source hash is `050ba61e3d018f643bf5c463e605f551da093f13172f287dba587a1d7c04d466`;
the canonical selection-lock hash is `6408c4737250c4bc0c0d27a28b6e325b4928ed38e7cfc49ec0a203c524d998b8`.

The benchmark reuses 14 scenes scored in earlier campaigns. Validation scores
belong to the pre-refit checkpoints. Scoring only the final checkpoint leaves
the separate effect of the six additional training scenes unmeasured.
