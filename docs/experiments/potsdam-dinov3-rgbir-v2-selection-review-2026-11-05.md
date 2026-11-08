# DINOv3 RGBIR selection: configuration A

**Status: SUPERSEDED** — Selection outcome is finalized in the 2026-11-06 selection summary.

A completed eight passes and 6,912 updates on 2026-11-05. The selected
checkpoint reached **93.0441% OA and 80.7498% six-class mIoU** on six validation
scenes (199,409,714 scored pixels), above all four historical C floors. B
started at 19:15:43 Singapore time.

A uses pretrained DINOv3-B with RGB and baseline augmentation (3,635,730
trainable parameters). The [campaign protocol](potsdam-dinov3-rgbir-v2.md)
defines selection; the [GPU
preflight](potsdam-dinov3-rgbir-v2-preflight-2026-11-05.md) explains the ViT-B
schedule.

## Validation and checkpoint selection

The two selection points use full-scene validation with 512 × 512 windows,
stride 256 and eroded labels.

| Checkpoint | OA | Six-class mIoU | Car IoU | Clutter IoU | Eligible |
| --- | ---: | ---: | ---: | ---: | --- |
| Historical C, seed 42 reference | 92.8978% | 79.5643% | 92.9585% | 38.9751% | Reference |
| A, update 3,456 | 92.5474% | 79.8985% | 93.6983% | 41.8505% | No: OA below floor |
| **A, update 6,912** | **93.0441%** | **80.7498%** | **93.9552%** | **43.8339%** | **Yes** |
| Selected A minus reference, percentage points | +0.1463 | +1.1854 | +0.9967 | +4.8588 | All floors passed |

Eligibility uses exact confusion counts. Selected checkpoint SHA-256:
`fbc022c284b42ed0a4a9a25397e9fa502267b549b2c10164b8a0b196fad066fb`.

The reference makes 14,162,511 validation errors; A makes 13,870,767, or
**291,744 fewer** (2.06%).

| Class | Selected A IoU | Selected A F1 |
| --- | ---: | ---: |
| Impervious surface | 88.7668% | 94.0492% |
| Building | 95.7587% | 97.8334% |
| Low vegetation | 80.3020% | 89.0750% |
| Tree | 81.8820% | 90.0386% |
| Car | 93.9552% | 96.8834% |
| Clutter/background | 43.8339% | 60.9507% |

Low vegetation predicted as tree accounts for 2,859,673 errors and the reverse
for 1,393,983 (**30.67%** combined). Clutter has the lowest IoU. B tests IR, the
spectral index, relief and DSM availability together. A and the historical
reference also differ in backbone, inputs, sampling and budget.

## Runtime and input diagnostics

A consumes **2.9335358 allocated GPU-hours**, counting both allocated T4s and
the full Kaggle execution log. Its worker-only receipt is 2.9290565 GPU-hours,
and the inner training runner records 2.9199813. Including both preflight
attempts, completed and verified phases total **3.3260785 GPU-hours**; the running
B phase is additional consumption and is not included in that subtotal.

Training takes 4,301.9162 seconds. Probes and full validation take 542.1253
seconds, **11.19%** of timed training plus validation or 10.31% of the full
inner runner. The older joint run spent 92.83% on validation under a different
schedule.

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

The local coordinator stopped while A continued on Kaggle. After Kaggle reported
COMPLETE, recovery downloaded A, checked the inventory, checkpoint,
source/protocol hashes and matrices, then submitted B.

The recovered coordinator ran under launchd as
`local.deepnet.potsdam-dinov3-rgbir-v2`. Its plist and logs are in the
campaign's `kaggle/` directory. A failed gate stops the job.

Primary artifacts and the derived review live under
`/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/runs/potsdam-dinov3-rgbir-v2/`:

- `study/A/manifest.json`, `best.pt`, `last.pt` and their artifact index.
- `kaggle/A/audit.json` and the original Kaggle execution log under `outputs/`.
- `review-selection/A-analysis.json`, class/scene CSVs and `A-review-index.json`.
- `kaggle/campaign-launchd.plist`, `campaign-launchd.log` and
  `campaign-launchd.err`.

Protocol SHA-256:
`db17701748df5596704e4da7f2b41d9f5c4d9cab0666e2475e81bfcdbd9edc7d`. Runner
source SHA-256:
`050ba61e3d018f643bf5c463e605f551da093f13172f287dba587a1d7c04d466`.
