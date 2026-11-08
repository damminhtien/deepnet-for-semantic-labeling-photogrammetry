# Potsdam DINOv2 coverage and refit artifact review, 2026-11-05

**Status: HISTORICAL** — Completed DINOv2 benchmark record retained for system-level context.

The 24-scene continuation completed its declared 1,152 optimizer updates, and
the final checkpoint was evaluated once on all 14 held-out scenes. Independently
recomputed OA is **91.4785%** and six-class mIoU is **79.6122%** on exactly
467,862,855 valid pixels. This exceeds historical SAM by 3.6643 OA points and
5.9097 mIoU points. The result establishes the measured single-model DINOv2
reference for the subsequent Potsdam study.

## Final held-out results

All three rows use the same 14 scene IDs, scored counts and aggregate class
supports. The audit verified historical metrics-file hashes before recomputing
scores. Byte equality of the historical reference maps remains unverified.

| Experiment | OA | Six-class mIoU |
| --- | ---: | ---: |
| Historical SAM ViT-B + LoRA | 87.8142% | 73.7024% |
| Historical joint DINOv2 ViT-B/14 + LoRA | 74.6576% | 48.6523% |
| Final Potsdam DINOv2 C + 24-scene refit | **91.4785%** | **79.6122%** |

The final model improves both OA and mIoU over SAM on all 14 scenes and IoU on
all six aggregate classes. Relative to the older joint DINOv2 recipe, the gains
are 16.8209 OA points and 30.9599 mIoU points. These are complete-recipe comparisons.

| Class | SAM IoU | Final IoU | IoU delta | Final F1 |
| --- | ---: | ---: | ---: | ---: |
| Impervious surfaces | 83.2068% | 87.8091% | +4.6023 pp | 93.5089% |
| Building | 86.2614% | 94.6559% | +8.3945 pp | 97.2546% |
| Low vegetation | 74.2724% | 79.0312% | +4.7588 pp | 88.2876% |
| Tree | 76.3332% | 81.0786% | +4.7455 pp | 89.5507% |
| Car | 85.1687% | 92.7234% | +7.5546 pp | 96.2243% |
| Clutter/background | 36.9718% | 42.3747% | +5.4029 pp | 59.5256% |

Clutter remains the weakest class. Its precision rises from 53.4148% to
70.8307%, while recall falls from 54.5667% to 51.3326%. Cars reach 94.9827%
precision and 97.4989% recall, up on both measures.

The final confusion matrix contains 39,868,882 misclassified pixels. Tree/low
vegetation confusions in both directions contribute 11,813,440 pixels, or 29.63%
of all errors. Another 4,876,846 clutter pixels become impervious surfaces and
2,572,988 become low vegetation. Scene `4_14` is the weakest at OA 84.5825% and
mIoU 72.7476%; OA across scenes ranges up to 94.1487%, and mIoU up to 84.7020%.

## Selection and continuation

Selection used 18 training scenes and six validation scenes, with four passes per
configuration and full validation only at updates 1,728 and 3,456. C won the
seed-42 ranking at update 3,456: OA 92.8978%, mIoU 79.5643%, car IoU 92.9585% and
clutter IoU 38.9751% on 199,409,714 validation pixels. D repeated C at seed 43
and passed the same four historical SAM floors. C's advantage over A was only
0.0434 OA points and 0.0436 mIoU points; the repeat validates eligibility but does
not establish a statistically reliable augmentation gain.

The selection lock records C's checkpoint SHA-256
`953d91bf89cfdb1add0755659632bf2b427c127bba311e68ef8f8a10bca6cd54` as the
starting point for refit. C uses IRRG, D4 and affine/optical augmentation. All
four selection runs and refit use DINOv2 ViT-B/14 with a residual adapter and a
normalized width-128 decoder.

The final C-plus-refit path records 4,575,287,363 valid Potsdam pixel
presentations, compared with 483,139,259 through the selected epoch 15 of the
older joint DINOv2 run, a factor of 9.47. These counts include repeated samples
and do not measure distinct coverage. The model also reduces trainable
parameters from 11,777,170 to 3,634,770 (69.14%). Exposure, decoder size,
normalization, adapter initialization, sampling and dataset scope all changed;
an accuracy difference cannot identify any one of them as its cause.

Refit used a separate train-only bundle with participant labels for all 24
scenes in the 18+6 union. Its fingerprint is
`a32734f4410dc97fa19b91d68bc49f8643390d7d4fc573bfd1ec37f5e3ee0679`.

AdamW and the AMP scaler were reset for refit. Initial learning rates were
5e-5 for LoRA and 1.5e-4 for the adapter/decoder, with 5% warmup and cosine
decay. One complete pass contains 3,456 base windows and 576 prioritized draws
for each of cars and clutter. Four microbatches per update yield 1,152 updates.
An independent reconstruction from the locked bundle metadata confirms 144
distinct base windows for each of the 24 scenes and complete 6,000 x 6,000
scene coverage; edge padding does not replace any interior pixel.
No validation, early stopping or checkpoint ranking ran during this stage.
The final weights at the declared final step are the evaluation model.

## Checkpoint and input audit

- Both `final.pt` and `last.pt` have the expected configuration/data binding,
  update 1,152 and one completed pass. The resumable state includes AdamW and
  the expected initial learning rates; all 178 AdamW tensor states are at step
  1,152, confirming the new phase's counter. Every floating model tensor is finite.
- All four files in the refit artifact inventory match their sizes and SHA-256
  values. Final checkpoint SHA-256:
  `03af62b805bad33c3ba29746c17406daeb772469229cb61d04af0e76d263d33d`.
- Comparing C with the final weights finds changes in all 96 LoRA tensors,
  eight adapter tensors and all 74 decoder tensors. All 223 non-LoRA backbone
  tensors remain exactly equal. This confirms continued optimization with the
  original backbone frozen.
- Refit recorded 1,144,117,429 valid pixel presentations. These include repeated
  rare-class sampling and are not a count of distinct pixels.
- The fixed 36-patch input panel contains 9,023,616 valid, non-ignore pixels per
  channel. Both projected and normalized views have zero nonfinite values and
  zero values outside the configured identity bounds. The final normalized
  channel means are -0.4756, -0.4613 and -0.3616; standard deviations are
  0.6123, 0.5750 and 0.5868. The complete manifest preserves min/max and all five
  requested percentiles. Masked and ignore pixels are excluded from this audit.
- The refit manifest records 1,137.985 seconds of optimizer-loop training and
  zero validation seconds. The phase budget records 0.8211 GPU-hours, charging
  both allocated T4 GPUs. The budget also includes work outside the timed
  training loop.

## Evaluation boundaries

The held-out kernel was submitted once at 13:26:47 SGT on 2026-11-05, as version
1 of [the private evaluation job](https://www.kaggle.com/code/damminhtien/potsdam-dinov2-coverage-refit-v1-heldout).
The prelaunch checks verified the source package, protocol, selection lock,
final refit checkpoint and unopened evaluation role. Inference uses 504 x 504
windows, stride 252 and weighted overlap on 14 scenes with eroded references.
The required support is exactly 467,862,855 pixels.

Kaggle completed version 1; the local audit finished at 13:48 SGT. The attempt
marker is bound to the final refit manifest, the metric record names the audited
final checkpoint and the evaluation count is exactly one. All three indexed
evaluation files match their hashes. All 14 per-scene matrices sum to the saved
aggregate; OA, mIoU, IoU, precision, recall and F1 were independently recomputed.

The canonical protocol hash is
`de5b58717cb3127c8d89c366d450ffca0dbc2ec7005fc5aeebc949b06b8f24b7`.
The runner source hash is
`3427b398bfc7d743a03ade4e8169f2dbf3911a57232abee030ef58ae7ceda7da`;
the source package SHA-256 is
`1378b92c6eed1cf6a12f83ab8555a15a26bb26d8aea7c89cc7759ef8a1bb80fe`.
These identities were frozen by the pilot and retained through selection,
confirmation, refit and evaluation.

Validation scores belong to the pre-refit checkpoint. The final checkpoint was
scored on the same historical cohort as prior runs. Comparisons cover complete
recipes with different training schedules and, for SAM, different inference
tiling. The effect of adding six scenes was not measured separately.

## Runtime and budget

| Phase | Runner-accounted GPU-hours |
| --- | ---: |
| Completed pilot | 0.2836 |
| A | 2.7075 |
| B | 3.4812 |
| C | 2.6159 |
| D, seed-43 confirmation | 2.7432 |
| 24-scene refit | 0.8211 |
| Single held-out evaluation | 0.6442 |
| **Total completed phases** | **13.2968** |

The remote execution logs, including bootstrap outside the phase timers, total
13.4594 GPU-hours across both T4s. The failed preflight adds 0.0103 GPU-hours
(18.461 wall-seconds), bringing the observed total to 13.4696 of 24. It stopped
at the allocation check. Validation occupied 24.28% / 36.51% / 24.37% / 24.30%
of timed training plus validation for A/B/C/D, versus 92.83% in the older joint
run.

## Evidence

External campaign root:
`/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/runs/potsdam-dinov2-coverage-refit-v1`.

- `kaggle/preflight-v2-evidence/outputs/study/`: immutable protocol, selection
  manifests and lock, plus the downloaded refit output.
- `kaggle/preflight-v2-evidence/audit-refit.json`: independent file, checkpoint,
  bundle, phase-completion and input-distribution checks.
- `kaggle/preflight-v2-evidence/audit-refit-weight-delta.json`: comparison with
  the selected C weights, including the frozen backbone check.
- `kaggle/preflight-v2-evidence/audit-refit-optimizer-reset.json`: AdamW step
  counters, initial/final learning rates and AMP scaler state.
- `kaggle/preflight-v2-evidence/audit-refit-grid-coverage.json`: reconstructed
  base grids and scene coverage for the real 24-scene bundle.
- `kaggle/preflight-v2-evidence/heldout-submission.log` and continuation logs:
  the single accepted evaluation submission and subsequent retrieval state.
- `kaggle/preflight-v2-evidence/audit-heldout.json`: the final checkpoint,
  attempt binding, three verified files and independently recomputed metrics.
- `review-final-20261105/analyze.py`: analysis of audited confusion matrices
  without inference. It verifies historical metric-file hashes and writes
  comparison tables, figures and an analysis inventory outside Git.
- `review-final-20261105/analysis.json`, `per-class.csv`, `per-scene.csv` and
  `confusion.csv`: complete metrics, comparisons, remaining errors and budgets.
- `review-final-20261105/per-class-iou.png` and `.pdf`: standalone comparison
  figures, rendered from the persisted counts. The PNG was visually inspected.
- `review-final-20261105/sha256.json`: hashes of the analysis, input evidence
  and generated tables/figures. No inference is run by this analysis.
