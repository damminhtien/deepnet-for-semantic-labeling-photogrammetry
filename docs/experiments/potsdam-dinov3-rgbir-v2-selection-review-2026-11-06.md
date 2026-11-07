# DINOv3 RGBIR selection: A and B

After eight passes with seed 42, **RGB-only A outranked B with IR, NDVI, relief
and DSM availability**. Both passed the four historical C floors on 199,409,714
validation pixels. C started at 13:28:03 Singapore time on 2026-11-06 with A's
RGB recipe, enhanced augmentation and fresh pretrained weights.

See the [protocol](potsdam-dinov3-rgbir-v2.md), [preflight
report](potsdam-dinov3-rgbir-v2-preflight-2026-11-05.md) and [A
review](potsdam-dinov3-rgbir-v2-selection-review-2026-11-05.md).

## Selection results

Full validation uses 512x512 windows, stride 256 and the same eroded labels.
Only updates 3,456 and 6,912 can select a checkpoint. Eligibility and ordering use
exact confusion matrices before rounding.

| Checkpoint | OA | Six-class mIoU | Car IoU | Clutter IoU | Eligible |
| --- | ---: | ---: | ---: | ---: | --- |
| Historical C seed 42 reference | 92.8978% | 79.5643% | 92.9585% | 38.9751% | Reference |
| A, update 6,912 | **93.0441%** | **80.7498%** | **93.9552%** | **43.8339%** | Yes |
| B, update 3,456 | 92.7130% | 80.1404% | 93.3572% | 42.4849% | No: OA below floor |
| B, update 6,912 | 92.9923% | 80.6097% | 93.9180% | 43.4670% | Yes |
| B minus A, percentage points | -0.0517 | -0.1401 | -0.0372 | -0.3669 | A ranks first |

B made **13,973,954 validation errors** versus A's 13,870,767 (103,187 more). A
ranked first and supplied C's recipe.

| Class | A IoU | B IoU | B minus A, points |
| --- | ---: | ---: | ---: |
| Impervious surface | 88.7668% | 88.7523% | -0.0145 |
| Building | 95.7587% | 95.6758% | -0.0829 |
| Low vegetation | 80.3020% | 80.1413% | -0.1607 |
| Tree | 81.8820% | 81.7036% | -0.1784 |
| Car | 93.9552% | 93.9180% | -0.0372 |
| Clutter/background | 43.8339% | 43.4670% | -0.3669 |

All six aggregate class IoUs are slightly lower in B. B improves scene OA on
`3_12` and `5_11`, while the other four scenes decline. The largest scene decline
is `4_12`, from 93.7644% to 93.3749% OA. The class and scene CSVs preserve the
unrounded values.

B's auxiliary gate moved from zero to -0.1867475 (`tanh(gate)` -0.1846064). A
and B share seed, sampling, baseline augmentation and budget, with 6,993,985,512
valid pixel presentations each. The comparison tests the combined auxiliary
features at seed 42.

## Runtime, inputs and evidence

| Measurement | A | B |
| --- | ---: | ---: |
| Optimizer updates | 6,912 | 6,912 |
| Trainable parameters | 3,635,730 | 3,655,603 |
| Timed training | 4,301.9162 s | 4,874.1140 s |
| Timed probes and full validation | 542.1253 s | 496.5227 s |
| Validation / (train + validation) | 11.19% | 9.25% |
| Full execution, both allocated T4s | 2.9335358 GPU-h | 3.2024464 GPU-h |

Preflight and A/B used **6.5285249 allocated GPU-hours**. Full execution
includes the Kaggle wrapper. B used 3.1987237 GPU-hours in the worker and
3.1922188 in the inner runner. The validation shares use a different schedule
from the older joint run's 92.83%.

B's 36-patch input panel remains finite at both full validations. At update
6,912 the projected RGB means are 0.334940/0.347924/0.319858. Nine of 9,068,544
red values exceed identity bounds; green and blue have none. The original
diagnostics retain moments, extrema and percentiles before/after normalization.

The review rechecks both phase artifact inventories, checkpoint hashes, the
frozen source/protocol binding, all **60 persisted train/probe/full-scene and
per-scene confusion matrices**, and each arm's valid-pixel exposure sum. B's
selected update-6,912 checkpoint SHA-256 is
`cfacfd341ccac5527f8622ea47149f116a372f79ed2ccff748dfa155dcbe1222`.
A's selected hash remains
`fbc022c284b42ed0a4a9a25397e9fa502267b549b2c10164b8a0b196fad066fb`.

All primary and derived artifacts remain outside Git under
`/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/runs/potsdam-dinov3-rgbir-v2/`.
`study/B/` holds the original weights, manifest and input diagnostics;
`kaggle/B/` holds the audit and provider execution log. The new
`review-selection-20261106/` directory holds the A/B analysis, class/scene CSVs
and output hash inventory. Earlier reports and artifact payloads are unchanged.

## Recovery across machine restarts

The first user launchd job was registered only for the current login session.
The machine restarted, so the job was absent when B was found COMPLETE. Recovery
retrieved and audited B, then installed a login LaunchAgent at
`~/Library/LaunchAgents/local.deepnet.potsdam-dinov3-rgbir-v2.plist` before
submitting C. Its program, working directory, PATH and log paths are explicit;
`KeepAlive` is false, so a failed gate does not trigger automatic retries.

After a second restart at **13:39:38 Singapore time on 2026-11-06**, launchd
loaded the agent as PID 1210 with parent PID 1. It resumed monitoring Kaggle's
existing C version 1.

`kaggle/coordinator-recovery-20261106.json` records the plist, checksum, boot
time, PID and C receipt. After the final audit, unload the service and remove
its plist only if the checksum matches.
