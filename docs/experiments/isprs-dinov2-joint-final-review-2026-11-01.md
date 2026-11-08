# DINOv2 joint-run final artifact review, 2026-11-01

**Status: HISTORICAL** — Completed separate joint Potsdam/Vaihingen protocol.

Potsdam held-out mIoU was **48.6523%** for joint DINOv2 versus **73.7024%** for
historical SAM, a 25.0501-point gap. DINOv2 scored lower on all six classes and
all 14 test scenes; cars and clutter had the largest losses. The comparison
covers complete training and evaluation recipes.

## Completion and artifact integrity

- Kernel: [damminhtien/dinov2-joint-irrg-segmentation-20261101](https://www.kaggle.com/code/damminhtien/dinov2-joint-irrg-segmentation-20261101), version 1, Tesla T4.
- Evaluation finished at 16:47:46 SGT; the Kaggle job exited at 16:47:53 SGT.
- Downloaded 12 indexed artifacts totaling 882,900,091 bytes, plus the artifact
  index, bootstrap/runtime records, source manifest and remote execution log.
  Both the supplied watcher audit and independent audit report no errors.
- Selected `best.pt`: epoch 15, optimizer step 960, SHA-256
  `22c5c4bd342b79cc767a2f8c8f4c03c73e09c60ee14d5803250b5573f0b033f9`.
  Its saved validation results equal `metrics.json` and it maximizes the recorded
  mean validation mIoU across all 20 epochs.
- `last.pt` contains epoch 20 / step 1,280, optimizer, AMP scaler and RNG state.
  The two checkpoints and manifest share the same data/model/configuration binding.
- The frozen checkpoint was scored once on each held-out dataset.
- Independently recomputed all 45 aggregate/per-scene confusion-matrix summaries:
  mIoU, OA, precision, recall and F1 match persisted values. Scene counts sum to
  each dataset's aggregate. Checkpoint, metric, configuration, protocol and source
  hashes match the saved launch identity; the frozen training config is unchanged.

After the local watcher stopped, retrieval resumed from the remote terminal
result. The download and audit finished at 17:03:45 SGT. The launch record and
final reconciliation are stored separately.

## Final metrics

All mIoUs cover six classes and ignore label 6. Validation scores use epoch 15.

| Dataset / split | Scenes | Scored pixels | mIoU | OA |
| --- | ---: | ---: | ---: | ---: |
| Potsdam validation | 6 | 199,409,714 | 51.0339% | 77.9859% |
| Potsdam test | 14 | 467,862,855 | 48.6523% | 74.6576% |
| Vaihingen validation | 4 | 17,591,104 | 47.6903% | 79.1191% |
| Vaihingen test | 17 | 81,991,943 | 49.1788% | 78.3033% |

Mean validation mIoU across both datasets was 49.3620805891%. Potsdam drops
2.3816 points from validation to test; most of its gap to SAM is already present
on validation.

## Potsdam comparison with historical SAM

| Test class | SAM IoU | DINOv2 IoU | DINOv2 delta |
| --- | ---: | ---: | ---: |
| Impervious surfaces | 83.2068% | 62.7431% | -20.4637 pp |
| Building | 86.2614% | 58.0959% | -28.1656 pp |
| Low vegetation | 74.2724% | 62.9809% | -11.2915 pp |
| Tree | 76.3332% | 65.9835% | -10.3496 pp |
| Car | 85.1687% | 39.9906% | -45.1781 pp |
| Clutter/background | 36.9718% | 2.1195% | -34.8523 pp |
| Six-class mean | 73.7024% | 48.6523% | -25.0501 pp |

SAM test OA was 87.8142% versus 74.6576% for DINOv2, a 13.1566-point gap. Cars
and clutter account for 13.3384 points (53.25%) of the mIoU gap. Excluding
clutter gives five-class diagnostic means of 57.9588% for DINOv2 and 81.0485%
for SAM.

Potsdam validation and test scene IDs, scored counts and class supports match
SAM; reference-map byte equality is unverified. DINOv2 test-scene mIoU ranges
from 43.02% to 52.65%, with gaps of 19.76–27.88 points versus SAM. SAM scored
non-overlapping 512-pixel tiles; DINOv2 used overlapping 504-pixel tiles.

## Failure modes supported by counts

- Potsdam clutter has 19,920,368 truth pixels but only 1,211,562 predicted pixels.
  Recall is 2.2018%, precision 36.2013%, and IoU 2.1195%. Of its true pixels,
  57.20% become impervious surfaces, 15.89% low vegetation and 14.54% building.
- Potsdam car recall is 68.0630% but precision is 49.2281%. The model predicts
  10,294,298 car pixels against 7,445,584 truth pixels. It misses 23.13% of true
  car pixels as building and also produces many false car predictions.
- Potsdam building loses 24.60% of its true pixels to impervious surfaces. Thus
  the regression extends beyond small or rare classes.
- Vaihingen car is also weak: IoU 21.3713%, recall 40.7919%, precision 30.9818%.
  Clutter IoU is 11.2618%, with recall 13.6872%.
- Vaihingen clutter IoU is 1.8781% on 5,634 validation truth pixels and 11.2618%
  on 692,551 test truth pixels.

Across 20 epochs, Potsdam contributed 644,140,612 valid semantic pixel
presentations out of 650,280,960 total, including repeats. Cars contributed
12,450,606 pixels (1.93%) in 1,386/2,560 patches; clutter contributed 44,341,795
pixels (6.88%) in 1,781/2,560 patches.

## Checkpoint findings and causal limits

All 48 LoRA B matrices are nonzero in the selected checkpoint, so adapter updates
occurred. No non-LoRA backbone tensor changes between best and last checkpoints.
The audit found no mismatch in checkpoint selection or metric arithmetic.

The selected spectral adapter's third channel is approximately:

$$
z_3 = -0.43493\,\mathrm{IR} - 0.45043\,R - 0.51341\,G + 0.00967
$$

The weights give analytical bounds of $[-1.3891, 0.0097]$ for the third channel on
IRRG inputs in $[0,1]$, and $[-7.9782, -1.7615]$ after mean/std normalization. Test
the accuracy effect with a controlled adapter-initialization experiment.

Other candidate explanations are unnormalized intermediate features, a larger
random decoder without refinement normalization, less Potsdam exposure and the
joint training objective.

Verify gradient flow and small-subset fit, then establish a Potsdam-only
baseline with reference feature normalization and channel-preserving
initialization. Match coverage and training budgets while varying one component
at a time. Test the effect of adding Vaihingen after that baseline.

## Local evidence and figures

External run root:
`/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/runs/isprs-dinov2-joint-v1-20261101-attempt-02`.

- Original downloaded core artifacts:
  `monitor/outputs/isprs-dinov2-joint-v1-20261101-attempt-02/`.
- Supplied artifact audit: `monitor/local-audit.json`.
- Independent checkpoint/count/provenance audit:
  `review-final-20261101/independent-audit.json`.
- Per-class and per-scene CSVs, actual training exposure and adapter weights/bounds
  are in `review-final-20261101/`.
- `review-final-20261101/analyze.py` regenerates the audits and Matplotlib figures
  `per-class-iou.png/.pdf` and `test-confusion.png/.pdf` from saved artifacts.
  Both PNGs were rendered and visually inspected. No inference is executed.
- `review-final-20261101/review-sha256.json` binds the analysis, raw inputs and
  rendered outputs. Historical SAM metrics remain in the preserved
  `sam-vitb-lora-kaggle-output-20261020/snapshot-20261020-retrain-04/` tree.

See the [earlier validation
review](isprs-dinov2-joint-validation-review-2026-11-01.md) for the learning
curve and exposure comparison.
