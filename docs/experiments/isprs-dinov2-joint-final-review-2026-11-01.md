# DINOv2 joint-run final artifact review, 2026-11-01

The private Kaggle attempt completed successfully. Both checkpoints, all metrics
and provenance are downloaded and independently verified. The selected DINOv2
recipe substantially underperforms historical SAM on Potsdam: held-out mIoU is
48.6523% versus 73.7024%, a 25.0501 percentage-point gap. Cars and clutter account
for the largest losses, but all six classes and all 14 Potsdam test scenes regress.
This is a comparison of full recipes, not an isolated backbone experiment.

## Completion and artifact integrity

- Kernel: [damminhtien/dinov2-joint-irrg-segmentation-20261101](https://www.kaggle.com/code/damminhtien/dinov2-joint-irrg-segmentation-20261101), version 1, Tesla T4.
- Kaggle status was freshly confirmed COMPLETE. Scientific evaluation completed
  at 16:47:46 SGT; the enclosing flow exited successfully at 16:47:53 SGT.
- Downloaded 12 indexed artifacts totaling 882,900,091 bytes, plus the artifact
  index, bootstrap/runtime records, source manifest and remote execution log.
  Both the supplied watcher audit and independent audit report no errors.
- Selected `best.pt`: epoch 15, optimizer step 960, SHA-256
  `22c5c4bd342b79cc767a2f8c8f4c03c73e09c60ee14d5803250b5573f0b033f9`.
  Its saved validation results equal `metrics.json` and it maximizes the recorded
  mean validation mIoU across all 20 epochs.
- `last.pt` contains epoch 20 / step 1,280, optimizer, AMP scaler and RNG state.
  The two checkpoints and manifest share the same data/model/configuration binding.
- Both held-out datasets were evaluated once using the frozen selected checkpoint.
  No local inference, test rerun, parameter change or new training was performed.
- Independently recomputed all 45 aggregate/per-scene confusion-matrix summaries:
  mIoU, OA, precision, recall and F1 match persisted values. Scene counts sum to
  each dataset's aggregate. Checkpoint, metric, configuration, protocol and source
  hashes match the saved launch identity; the frozen training config is unchanged.

The previous local watcher stopped before completion. Retrieval was restarted
after observing the remote terminal state; download and audit completed at
17:03:45 SGT. The original launch record is preserved, including its historical
RUNNING status; final reconciliation is stored separately.

## Final metrics

All mIoUs below include six semantic classes, with label 6 ignored. Validation
values belong to selected epoch 15, not the last epoch.

| Dataset / split | Scenes | Scored pixels | mIoU | OA |
| --- | ---: | ---: | ---: | ---: |
| Potsdam validation | 6 | 199,409,714 | 51.0339% | 77.9859% |
| Potsdam test | 14 | 467,862,855 | 48.6523% | 74.6576% |
| Vaihingen validation | 4 | 17,591,104 | 47.6903% | 79.1191% |
| Vaihingen test | 17 | 81,991,943 | 49.1788% | 78.3033% |

The validation selection mean is 49.3620805891%. Potsdam loses only 2.3816 points
from validation to test; its much larger gap to SAM is already present on
validation. These observations do not establish overfitting or underfitting
without training-set accuracy and controlled experiments.

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

Historical SAM test OA is 87.8142%, versus 74.6576% now, a 13.1566-point gap.
Cars and clutter contribute 13.3384 points, or 53.25%, of the total mIoU gap.
Excluding clutter for diagnosis still leaves five-class mean IoU at 57.9588%
versus SAM's 81.0485%; that diagnostic average does not replace the six-class score.

Potsdam scene IDs, per-scene scored counts and aggregate ground-truth class
supports exactly match SAM for both validation and test. DINOv2 is lower on every
test scene, with scene mIoU ranging from 43.02% to 52.65%; the paired gaps range
from 19.76 to 27.88 points. This is not one anomalous test tile. Historical and
current label-map byte equality was not established. Historical SAM also used
non-overlapping 512-pixel inference, while DINOv2 uses weighted-overlap 504-pixel
inference, so the comparison does not isolate architecture effects.

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
- Vaihingen validation contains only 5,634 clutter truth pixels, versus 692,551
  on test. Its clutter IoUs of 1.8781% and 11.2618% come from different class
  supports; a higher test mIoU does not by itself establish stronger generalization.

The epoch histograms rule out complete absence of these classes from training.
Across all 20 epochs, Potsdam supplied 644,140,612 valid semantic pixel
presentations out of 650,280,960 total. Cars contributed 12,450,606 pixels (1.93%)
in 1,386/2,560 patches; clutter contributed 44,341,795 pixels (6.88%) in
1,781/2,560 patches. These are presentations, including repeats, not unique pixels
or complete spatial coverage. More augmentation alone is not an established fix.

## Checkpoint findings and causal limits

All 48 LoRA B matrices are nonzero in the selected checkpoint, so adapter updates
occurred. No non-LoRA backbone tensor changes between best and last checkpoints.
The audit found no mismatch in checkpoint selection or metric arithmetic.

The selected spectral adapter exposes a concrete input-interface concern. Its
third projected channel is approximately:

```text
z3 = -0.43493 * IR - 0.45043 * R - 0.51341 * G + 0.00967
```

For unaugmented IRRG inputs scaled to [0,1], its exact analytical output bounds
are approximately [-1.3891, 0.0097]. After the configured mean/std normalization,
the third encoder channel lies in [-7.9782, -1.7615]. These are bounds derived from
weights, not a measured activation distribution. They demonstrate a strong shift
from the intended unit-range optical input before normalization. Its accuracy
effect still needs a controlled adapter-initialization experiment.

The earlier source review also found unnormalized intermediate hidden states,
a larger randomly initialized convolutional decoder without refinement
normalization, less Potsdam exposure and a different joint-domain objective.
Those remain hypotheses rather than separately measured causes. The current
evidence does not show that ViT-B/14 itself is the limiting choice.

For a new validation-controlled experiment, first verify the encoder/adapter
interface and small-training-subset fitting; establish a Potsdam-only baseline
with reference feature normalization and deterministic channel-preserving
initialization. Isolate individual changes, use a declared spatial-coverage and
optimization budget, then measure the effect of adding Vaihingen. Increasing the
backbone or adding more augmentations before these checks is not justified by
the present evidence. Preserve this run and its test observations as history.

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

The [earlier validation review](isprs-dinov2-joint-validation-review-2026-11-01.md)
remains an explicitly intermediate observation; the current document records the
final held-out result and verified completion.
