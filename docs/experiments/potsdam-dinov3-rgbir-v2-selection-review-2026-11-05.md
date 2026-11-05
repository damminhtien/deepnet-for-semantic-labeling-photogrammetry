# DINOv3 RGBIR selection: configuration A

Configuration A completed all eight passes and 6,912 optimizer updates on
2026-11-05. Its selected checkpoint reaches **93.0441% OA and 80.7498% six-class
mIoU** on the six validation scenes and exactly **199,409,714 scored pixels**.
It passes all four eligibility floors from historical C seed 42. Configuration
B was submitted once at 19:15:43 Singapore time and is running; C and seed-43
confirmation remain pending. No new benchmark evaluation has been opened.

This is an intermediate review of the [frozen campaign](potsdam-dinov3-rgbir-v2.md),
not a final experiment result. A uses DINOv3-B, verified RGB and baseline
augmentation, initialized from pretrained weights. It has 3,635,730 trainable
parameters. The [GPU preflight](potsdam-dinov3-rgbir-v2-preflight-2026-11-05.md)
records why the common eight-pass B schedule was admitted and satellite L excluded.

## Validation and checkpoint selection

Only the two predeclared full-scene validation points can choose a checkpoint.
Both use 512x512 windows, 256x256 stride and eroded labels. The 72-window probes
do not participate in checkpoint selection.

| Checkpoint | OA | Six-class mIoU | Car IoU | Clutter IoU | Eligible |
| --- | ---: | ---: | ---: | ---: | --- |
| Historical C, seed 42 reference | 92.8978% | 79.5643% | 92.9585% | 38.9751% | Reference |
| A, update 3,456 | 92.5474% | 79.8985% | 93.6983% | 41.8505% | No: OA below floor |
| **A, update 6,912** | **93.0441%** | **80.7498%** | **93.9552%** | **43.8339%** | **Yes** |
| Selected A minus reference, percentage points | +0.1463 | +1.1854 | +0.9967 | +4.8588 | All floors passed |

The eligibility decision uses exact confusion-matrix counts, not these rounded
values. The selected checkpoint's SHA-256 is
`fbc022c284b42ed0a4a9a25397e9fa502267b549b2c10164b8a0b196fad066fb`.

The reference makes 14,162,511 errors; A makes 13,870,767, or **291,744 fewer
errors, a 2.06% reduction**. Its validation OA remains 5.9559 percentage points
below 99%. On this validation support, reaching 99% would require at least
11,876,670 additional corrections. These are validation quantities; the previous
91.4785% post-refit benchmark OA uses a different scene set and is not a direct
baseline for A's validation OA.

| Class | Selected A IoU | Selected A F1 |
| --- | ---: | ---: |
| Impervious surface | 88.7668% | 94.0492% |
| Building | 95.7587% | 97.8334% |
| Low vegetation | 80.3020% | 89.0750% |
| Tree | 81.8820% | 90.0386% |
| Car | 93.9552% | 96.8834% |
| Clutter/background | 43.8339% | 60.9507% |

Low vegetation predicted as tree accounts for 2,859,673 errors, and the reverse
for 1,393,983. Together they account for **30.67% of A's errors**. Clutter remains
the weakest class. B will test the joint addition of IR, the spectral index,
relief and DSM availability; it cannot isolate the contribution of each feature.
The A/reference comparison also changes backbone, inputs, sampling and training
budget, so it does not establish a backbone-only improvement.

## Runtime and input diagnostics

A consumes **2.9335358 allocated GPU-hours**, counting both allocated T4s and
the full Kaggle execution log. Its worker-only receipt is 2.9290565 GPU-hours,
and the inner training runner records 2.9199813. Including both preflight
attempts, completed and verified phases total **3.3260785 GPU-hours**; the running
B phase is additional consumption and is not included in that subtotal.

Timed training takes 4,301.9162 seconds, and probes plus full validation take
542.1253 seconds. Validation is therefore **11.19% of timed training plus
validation**, compared with the historical 92.83% in the older joint run. Using
the entire inner-runner duration as denominator gives 10.31%. These scopes and
the changed training/validation schedules must accompany the comparison; the
figures do not represent a controlled hardware speedup experiment.

All 2,592 base windows are visited in each pass, with 432 car and 432 clutter
draws. Eight passes record **6,993,985,512 valid pixel presentations after
augmentation**. The audit recomputes all 30 persisted train/probe/full-scene and
per-scene confusion matrices and verifies the exposure sum against the manifest.

The fixed 36-patch input diagnostics at both full validations contain zero
nonfinite values. At update 6,912, projected RGB means are
0.335076/0.345959/0.319268. Nine of 9,068,544 red-channel values exceed the
identity bounds, a fraction of 0.0000009924; green and blue have no exceedances.
The red maximum is 1.010597. The saved diagnostics retain standard deviations,
extrema and all requested percentiles before and after normalization.

## Execution continuity and evidence

The earlier chat-owned local coordinator stopped while Kaggle A continued to
completion. Its launch receipt and logs remain preserved. After a fresh remote
COMPLETE check, the recovery downloaded A once, verified its artifact inventory,
checkpoint progress/finite tensors, source/protocol binding and matrices, then
continued with B. No completed training phase was resubmitted.

The coordinator now runs as the user launchd job
`local.deepnet.potsdam-dinov3-rgbir-v2`, with launchd as its parent. Its plist,
stdout and stderr are outside Git in the campaign's `kaggle/` directory. The job
does not restart itself after a failed gate. This separates coordinator lifetime
from the chat's subprocess lifetime while preserving the existing submission
receipts and budget checks.

Primary artifacts and the derived review live under
`/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/runs/potsdam-dinov3-rgbir-v2/`:

- `study/A/manifest.json`, `best.pt`, `last.pt` and their artifact index.
- `kaggle/A/audit.json` and the original Kaggle execution log under `outputs/`.
- `review-selection/A-analysis.json`, class/scene CSVs and `A-review-index.json`.
- `kaggle/campaign-launchd.plist`, `campaign-launchd.log` and
  `campaign-launchd.err`.

The frozen protocol hash remains
`db17701748df5596704e4da7f2b41d9f5c4d9cab0666e2475e81bfcdbd9edc7d`,
and the runner source hash remains
`050ba61e3d018f643bf5c463e605f551da093f13172f287dba587a1d7c04d466`.
Refit still requires an eligible winner and eligible seed 43. Benchmark scoring
still requires the complete update-1,152 refit checkpoint and its locked hash.
