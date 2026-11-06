# DINOv3 RGBIR selection: A and B

Both eight-pass, seed-42 selection runs are complete and independently verified.
**RGB-only A remains ahead of B with IR, NDVI, relief and DSM availability.**
Both pass the four historical C eligibility floors on exactly 199,409,714
validation pixels. C started once on 2026-11-06 at 13:28:03 Singapore time with
A's RGB recipe and enhanced augmentation, initialized again from pretrained.
Its remote log reaches update 1,408/6,912 at 13:45:46. C's full validation,
seed-43 confirmation, refit and benchmark remain pending at this review.

The [frozen campaign protocol](potsdam-dinov3-rgbir-v2.md),
[preflight report](potsdam-dinov3-rgbir-v2-preflight-2026-11-05.md) and
[earlier A review](potsdam-dinov3-rgbir-v2-selection-review-2026-11-05.md)
retain the schedule, input provenance and previous observations. No new benchmark
has been opened; the old post-refit benchmark OA of 91.4785% uses different scenes
and must not be compared directly with these validation OAs.

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

B's selected checkpoint makes **13,973,954 errors**, versus A's 13,870,767:
103,187 additional errors. A's OA advantage is small, but the declared ranking
selects A as C's parent. The campaign does not change this ranking after seeing
the result. A remains 5.9559 OA percentage points below the 99% research target.

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

B's auxiliary gate moved from zero to raw -0.1867475, with effective
`tanh(gate)` -0.1846064. The auxiliary branch therefore participates in the
trained model. Its negative sign scales a learned feature representation; it
does not by itself identify a particular modality as harmful. A/B share seed,
sampling, baseline augmentation and training budget, and record the identical
6,993,985,512 valid-pixel presentations. The group of auxiliary features does
not improve this paired run. One seed does not establish a stable general
effect, and B does not isolate IR, NDVI or DSM individually.

C tests enhanced augmentation on the selected RGB recipe. Seed 43 will repeat
the eventual winner from pretrained. Both seeds must pass the original four
floors before refit. No feature or augmentation changes are made in response to
these intermediate results.

## Runtime, inputs and evidence

| Measurement | A | B |
| --- | ---: | ---: |
| Optimizer updates | 6,912 | 6,912 |
| Trainable parameters | 3,635,730 | 3,655,603 |
| Timed training | 4,301.9162 s | 4,874.1140 s |
| Timed probes and full validation | 542.1253 s | 496.5227 s |
| Validation / (train + validation) | 11.19% | 9.25% |
| Full execution, both allocated T4s | 2.9335358 GPU-h | 3.2024464 GPU-h |

Completed preflight attempts plus A/B total **6.5285249 allocated GPU-hours**.
C is running and is additional consumption. Full execution includes the Kaggle
notebook wrapper; B's worker-only receipt is 3.1987237 GPU-hours and its inner
runner records 3.1922188. These are execution scopes, not billing exports. The
validation shares accompany their different schedules when compared with the
historical 92.83% joint-run share; they are not a controlled hardware comparison.

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

A second restart at **13:39:38 Singapore time on 2026-11-06** exercised this
recovery path. The LaunchAgent loaded automatically, with PID 1210 and parent
PID 1, and resumed watching the existing C version 1. C continued running on
Kaggle; this was coordinator recovery, not checkpoint-based trainer resume.
No completed arm or existing C job was resubmitted.

The startup plist, its checksum, boot time, PID and C receipt are recorded in
`kaggle/coordinator-recovery-20261106.json`. Earlier control-plane receipts/logs
are preserved. After the terminal campaign audit, unload the task-owned service
and remove its login plist only if its checksum still matches this receipt.
The source and protocol hashes remain unchanged; benchmark access still requires
both seeds to qualify and the final update-1,152 refit checkpoint to verify.
