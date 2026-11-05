# DINOv3 RGBIR: verified GPU preflight

The second private Kaggle preflight completed and was independently audited on
2026-11-05. It admits **DINOv3-B for eight selection passes**. Satellite L passes
the model/quality gates but exceeds the selection budget. Configuration A is
running from fresh pretrained weights; there is no new full-validation or
benchmark accuracy yet. The [campaign protocol](potsdam-dinov3-rgbir-v2.md)
remains the source of selection, refit and benchmark rules.

## Quality and schedule

| Pilot measurement | ViT-B/16 | ViT-L/16 satellite |
| --- | ---: | ---: |
| LoRA B matrices with finite nonzero gradients | 48 | 96 |
| Fixed eight-patch fit: updates to pass | 64 | 32 |
| Train mIoU at that update | 95.3932% | 90.9492% |
| Median optimizer update, four microbatches | 0.6629 s | 1.3277 s |
| Fixed 72-window validation probe | 3.6090 s | 7.0147 s |
| Scene `4_12`, 512 window / 256 stride | 28.2614 s | 56.8200 s |
| Same scene with horizontal-flip TTA | 49.9589 s | 103.6927 s |
| Peak PyTorch allocated CUDA bytes | 845,088,256 | 1,950,844,928 |

The small fit uses the same eight fixed train patches, covers all six classes and
disables augmentation. The pilot discards these weights before selection. Both
backbones satisfy the >=90% train mIoU gate within 512 updates. These fit scores
establish learnability on eight patches, not validation quality or a ranking of
the two backbones. Peak tensor allocations are not total board-memory readings.

The budget forecast includes training, probes, two full validations, a TTA
allowance, measured setup and **25% headroom**, charging both allocated T4s:

| Proposed schedule | Forecast allocated GPU-hours per run | Applicable ceiling | Decision |
| --- | ---: | ---: | --- |
| B, 12 passes | 5.3072 | 4.0 without L | Reject |
| B, 8 passes | 3.7062 | 4.0 without L | Accept |
| L, 6 passes | 5.7650 | 3.2 with L | Reject |

The frozen roster is A/B/C plus seed-43 confirmation, all using B and eight
passes. Each pass has 864 updates; each run therefore has **6,912 updates**.
Only full validation at passes 4 and 8, or updates **3,456 and 6,912**, can choose
a checkpoint. The 72-window probe runs after every pass. L is excluded by the
predeclared cost gate, not by missing weights, insufficient VRAM or failed fit.

## Cache equivalence and profile

Cached and Rasterio inference produce **exactly the same confusion matrix** on
all **33,463,460 scored pixels** of validation scene `4_12`. Image, label and mask
patch checks also pass, including a padded edge. The scene-level cache is
**3.1674 times faster** end to end: 28.2614 versus 89.5161 seconds.

| Stage, same B model and scene | Memmap cache | Rasterio reference |
| --- | ---: | ---: |
| Source I/O | 0.4628 s | 22.1559 s |
| Preprocessing | 1.6860 s | 39.0724 s |
| Host to device | 1.5572 s | 1.5989 s |
| Model forward | 18.0342 s | 18.3861 s |
| Device to host | 0.7377 s | 1.0073 s |
| Memmap accumulation | 3.2035 s | 4.6665 s |
| Metric reduction | 2.0437 s | 2.0857 s |

The split source-I/O/preprocessing timers sit inside the evaluator's combined
read/preprocessing stage; small differences reflect wrapper and timer overhead.
Memmap initialization takes about 0.43 seconds on each path. Forward latency is
similar, while source I/O and preprocessing fall from about 61.23 to 2.15 seconds.
This identifies the source-reader path as the main removed cost in this panel.
It does not establish the same ratio on every scene or during training.

Non-overlap profiling takes 9.1439 seconds on the cached scene. The locked 50%
overlap policy costs **3.0907 times** as much. Its cost is included in the frozen
schedule. The pilot projects roughly 7.4% validation share for an eight-pass B
run; the actual training/validation share must come from each completed arm's
manifest before comparing it with the historical 92.83%.

The fixed 36-patch input panel contains 9,068,544 scored pixels per channel. RGB
means at initialization are 0.329278 / 0.349322 / 0.320644. Both models have zero
nonfinite values and zero fractions beyond the identity bounds before and after
backbone normalization, both at initialization and after their small fits.
The raw reports retain all requested moments, extrema and percentiles. Relief's
train-only absolute p99 scale is 5.766376972198486, unchanged across cache roles.

## Evidence and accounting

Artifacts reside outside Git under
`/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/runs/potsdam-dinov3-rgbir-v2/`.
The audit verifies the worker output inventory, report and budget hashes,
protocol/source bindings, train-index hash and successful B fit before submitting A.

- Frozen protocol hash:
  `db17701748df5596704e4da7f2b41d9f5c4d9cab0666e2475e81bfcdbd9edc7d`.
- Runner source hash:
  `050ba61e3d018f643bf5c463e605f551da093f13172f287dba587a1d7c04d466`.
- Selection cache hash:
  `49efbfb559d47932942924dc859e87116f0303c5c3629acd7215760cb02fb734`.

Preflight v1 failed before model loading because the source package omitted a
loss-module dependency. Its artifacts are preserved; the isolated archive-import
regression now covers that boundary. Kaggle execution logs charge **0.0150355
allocated GPU-hours** for v1 and **0.3775072** for v2, totaling **0.3925426**.
This includes the notebook wrapper after the worker exits. Worker-only receipts
total 0.3871520 GPU-hours; those original receipts are retained separately.
The in-run cumulative budget records 0.3848459 GPU-hours before its own final
artifact work. These are distinct execution scopes, not provider billing exports.

The new 24-hour campaign has not opened the benchmark. Selection must strictly
improve OA and preserve or improve mIoU/car/clutter IoU against the old C matrix;
seed 43 must confirm,
and the 24-scene refit must finish update 1,152 before the single benchmark attempt.
