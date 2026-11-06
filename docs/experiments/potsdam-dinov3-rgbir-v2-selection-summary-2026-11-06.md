# DINOv3 RGB selection and seed-43 confirmation

All three seed-42 runs have completed eight passes and 6,912 optimizer updates.
**A, the RGB baseline, wins the declared ranking.** A/B/C each pass all four
historical C floors on exactly 199,409,714 validation pixels. **D confirms A's
recipe with seed 43**, also passing all four floors. D started once on 2026-11-06
at 14:55:24 Singapore time, and its terminal artifacts were audited at 16:33:38.
The inference policy is locked to identity: both horizontal-flip OA gains are
below the declared 0.1-point requirement. The coordinator is preparing the
24-scene refit cache; the new benchmark remains closed at this review.

The [campaign protocol](potsdam-dinov3-rgbir-v2.md) and
[preflight](potsdam-dinov3-rgbir-v2-preflight-2026-11-05.md) retain the frozen
schedule and model roster. The earlier [A review](potsdam-dinov3-rgbir-v2-selection-review-2026-11-05.md)
and [A/B review](potsdam-dinov3-rgbir-v2-selection-review-2026-11-06.md) preserve
the preceding measurements and recovery evidence.

## Results and interpretation

The table reports each arm's selected update-6,912 checkpoint. Full-scene
validation uses 512x512 windows, stride 256 and eroded labels. The fixed probes
cannot choose a checkpoint.

| Configuration | OA | Six-class mIoU | Car IoU | Clutter IoU |
| --- | ---: | ---: | ---: | ---: |
| Historical C seed-42 reference | 92.8978% | 79.5643% | 92.9585% | 38.9751% |
| **A: RGB, baseline augmentation** | **93.0441%** | **80.7498%** | 93.9552% | **43.8339%** |
| B: RGB + IR/index/DSM auxiliary branch | 92.9923% | 80.6097% | 93.9180% | 43.4670% |
| C: A's RGB recipe, enhanced augmentation | 92.9690% | 80.5115% | **94.0595%** | 42.6426% |
| C minus A, percentage points | -0.0750 | -0.2382 | +0.1043 | -1.1913 |
| D: A's recipe, seed 43 confirmation | 92.9477% | 80.4596% | 93.7907% | 42.8664% |

![Seed-42 full validation and IoU differences from A](figures/potsdam-dinov3-rgbir-v2-abc-2026-11-06.png)

The left panel contains only the two scheduled full validations, passes 4 and 8;
the lines connect measured checkpoints. All pass-four OAs are below the floor,
and all pass-eight OAs exceed it. The right panel shows class IoU changes from
A. The PNG was visually inspected; its source PNG/PDF and input hashes remain
with the external review artifacts.

C improves car IoU by 0.1043 point but loses 1.1913 clutter IoU points. It makes
14,020,417 validation errors, versus 13,870,767 for A: **149,650 more errors**.
Its full pass-four checkpoint also fails the OA floor (92.4244%). Under the
locked eight-pass budget, the enhanced augmentation recipe does not improve
OA or mIoU in seed 42. This does not identify an individual augmentation as the
cause or establish the effect across seeds.

C's pass-eight train mIoU is 83.0787%, compared with 83.1445% for A, and its
mean training loss is 0.2344 versus 0.2329. These train quantities are measured
under each augmentation recipe and sampled class mix; they are not measurements
on a common unaugmented training panel. They do not alone explain the validation
difference or prove that more optimization would reverse it.

All arms cover the same 2,592 base windows and 432+432 rare draws per pass.
C records 6,991,404,308 actual valid-pixel presentations, versus 6,993,985,512
for A/B. The 2,581,204 difference is less than 0.04% and reflects the changed
augmentation path; C does not omit a train scene or a base coverage window.

A remains 5.9559 validation OA percentage points below 99%. These validation
scores cannot be compared directly with the previous post-refit 91.4785%
benchmark OA on different scenes. No new benchmark result exists yet.

## Confirmation and scoring gates

D repeats A from pretrained with the same eight-pass schedule. Its selected
checkpoint is update 6,912, with 14,062,959 validation errors. Compared with A,
OA is lower by 0.0964 point and mIoU by 0.2902 point. Both seeds clear the
predeclared floors; two seeds do not establish the full seed-to-seed variance.

Horizontal-flip TTA improves all four metrics for both checkpoints, but fails
the minimum OA-gain rule:

| Seed | Identity OA | Horizontal-flip OA | Gain, percentage points | Required |
| --- | ---: | ---: | ---: | ---: |
| 42 | 93.0441% | 93.1227% | +0.0786 | >=0.1000 |
| 43 | 92.9477% | 93.0167% | +0.0690 | >=0.1000 |

The frozen policy therefore remains **identity**, with 512x512 windows and
stride 256. The seed-42 checkpoint SHA-256 is
`fbc022c284b42ed0a4a9a25397e9fa502267b549b2c10164b8a0b196fad066fb`;
the confirmation checkpoint is
`7dc88b20879b950af86e0fb1665c9b790d1338530ea7306207ab46541c7c341a`.
The canonical selection-lock hash is
`6408c4737250c4bc0c0d27a28b6e325b4928ed38e7cfc49ec0a203c524d998b8`.
It fixes A's RGB/baseline-augmentation recipe, the selected seed-42 weights,
inference policy, half learning rates and exact 1,152-update refit. Only the
complete final checkpoint can open the single benchmark attempt. No selection
threshold or schedule was changed after observing the results.

## Accounting and audit

| Measurement | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| Timed training, seconds | 4,301.9162 | 4,874.1140 | 4,355.6156 | 4,335.0447 |
| Timed validation, seconds | 542.1253 | 496.5227 | 394.4229 | 470.9911 |
| Validation / (train + validation) | 11.19% | 9.25% | 8.30% | 9.80% |
| Full allocated GPU-hours | 2.9335358 | 3.2024464 | 2.8305651 | 3.2084227 |

Both allocated T4s count. The completed preflight attempts and A/B/C/D total
**12.5675127 allocated GPU-hours**, including the failed first preflight and
D's TTA. The TTA inner ledger records 0.3278253 GPU-hours, already contained in
D's full execution cost; it must not be added again.
C's worker-only scope is 2.8271300 GPU-hours and its inner runner records
2.8209043. These are execution-ledger scopes, not provider billing exports.
Validation now occupies a much smaller share than the old 92.83% joint-run
measurement, with the changed schedules and denominators recorded here. The
validation timer includes probes, full-scene scoring, the fixed input diagnostics
and selected-checkpoint writes; it is broader than model-forward latency.

The independent review verifies all four phase artifact inventories and
checkpoint hashes, all **134 saved train/probe/full-scene, TTA and per-scene
confusion matrices**, each arm's exposure sum, finite input panels and the
lock's seed/checkpoint/recipe bindings. There are 120 history matrices and 14
TTA matrices. D has 6,994,045,832 valid-pixel presentations. Its final 36-patch
input panel has no nonfinite values before or after normalization.
C's selected checkpoint SHA-256 is
`4d5bf3453a8466fb4d47869dd5066a1ffd677c05c2493561a16b51176eda6d58`.
The protocol remains
`db17701748df5596704e4da7f2b41d9f5c4d9cab0666e2475e81bfcdbd9edc7d`;
the source remains
`050ba61e3d018f643bf5c463e605f551da093f13172f287dba587a1d7c04d466`.

Primary weights/logs and derived analysis remain outside Git under
`/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/runs/potsdam-dinov3-rgbir-v2/`.
`study/C/`, `study/D/` and their `kaggle/` directories hold original outputs and
audits. `study/selection-lock.json` binds the decision; TTA matrices are retained
even though that policy was rejected. The directory
`review-selection-abc-20261106/` holds the ABC analysis, class/scene CSVs,
comparison PNG/PDF and their inventories. Only the small report PNG is copied
into Git, with identical bytes. `review-confirmation-20261106/` adds the
independent confirmation/TTA analysis, comparison CSV and inventory. Historical
artifacts are unchanged.
