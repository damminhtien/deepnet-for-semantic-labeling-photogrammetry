# Potsdam DINOv3 RGBIR v2

This campaign implements the protocol approved on 2026-11-05. It has a new
24 allocated-GPU-hour Kaggle budget. The previous final DINOv2 model scored
91.4785% OA and 79.6122% six-class mIoU. Reaching 99% OA on that support would
require about 88.26% fewer errors. This is a research target, not an acceptance
claim about the new code.

The implementation, offline checks and [CUDA preflight](potsdam-dinov3-rgbir-v2-preflight-2026-11-05.md)
are complete. The frozen schedule uses ViT-B for eight passes per run; L passes
the quality gates but exceeds the six-pass cost limit. Configuration A has
completed and passed all four validation floors; B is running. The
[selection review](potsdam-dinov3-rgbir-v2-selection-review-2026-11-05.md) records
A's verified metrics and accounting.
Selection, confirmation, refit and benchmark results must be read from the new
campaign's audited artifacts. Offline smoke and small-fit tests establish no
validation or benchmark accuracy.

## Sources and the historical channel discrepancy

The separate [RGBIR DVC snapshot](../../data/auxiliary/potsdam-rgbir-v2.dvc)
contains 38 four-band orthophotos, world files, the nested source archive and its
audit manifest. The official [ISPRS source description](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)
declares R/G/B/IR order. The snapshot builder checks archive/raster checksums,
uint8 band layout, CRS, affine, extent, mask coverage and scene IDs. It compares
every RGB pixel with the independent `2_Ortho_RGB.zip` reference before accepting
the four-band source.

On the present source files, all 38 historical three-band IRRG rasters match
RGBIR bands **4/2/3**, whereas the previously declared IR/red/green order would
require **4/1/2**. The new RGBIR manifest records this measured correspondence.
Historical source files, catalogs, trained weights and scores are preserved.
Interpret historical IRRG and spectral-index experiments with this discrepancy
in mind; their configured channel names do not establish the measured band order.
The new encoder receives verified R/G/B and its auxiliary index uses bands 4/1.

DSM uses the [existing verified snapshot](../../data/auxiliary/potsdam-dsm-v1.dvc)
and `valid-box-101-p99abs-stride16-v1` relief recipe. Relief subtracts the valid
101x101 local mean. Its absolute p99 scale is fitted on the original 18 training
scenes, then applied unchanged to every role and clipped to [-1,1]. Scene `3_13`
retains its missing last DSM column with zero relief and availability. DSM
availability never intersects the image/label scoring mask. JPEG nDSM stays
blocked pending registration and encoding evidence.

## Caches and augmentation

`deepnet.potsdam_rgbir.prepare` builds separate selection, refit and benchmark
caches. Each scene stores read-only memory-mapped RGBIR uint8, relief float32,
DSM availability uint8, class ID uint8 and validity uint8 arrays. The manifest
binds original bundle, split, RGBIR snapshot, DSM source, recipe, scale, label
role and every array's bytes/hash. Refit accepts exactly the historical 18+6
union and participant labels, with no validation/test role. Reference readers
reject changed hashes, shapes, roles and unrecognized source mappings.

The selection cache has 24 scenes, 123 inventoried files and 9,504,068,769 bytes.
The local audit confirms 199,409,714 scored validation pixels and exact
image/label/mask equality with Rasterio on 12 fixed reference windows. The GPU
pilot additionally requires identical full-scene confusion matrices for cached
and Rasterio inference on scene `4_12`.

Each pass covers all 2,592 original 512x512 windows, including padded edge
windows. It adds 432 car and 432 clutter windows. Rare sampling first chooses an
eligible training scene, then uniformly selects a target-class pixel from all
its training pixels using tile counts, then jitters the center by at most 96
pixels. There is no rare-pixel reservoir. Four microbatches give 864 optimizer
updates. The index records original source coverage and class support; training
records actual valid pixel presentations after augmentation. The 24-scene refit
uses 3,456 base windows plus 576+576 rare draws, or 1,152 updates.

Two loader workers prefetch two batches with persistent workers and CUDA pinned
memory. Per-draw seeds bind study seed, pass and draw ordinal. A separate loader
generator preserves the restored model RNG when resuming. Geometry is shared by
all bands/labels/masks; labels and masks use nearest interpolation and continuous
features use validity-weighted interpolation. Both recipes use D4. Affine is
disabled for rare draws. The model computes IR/red NDVI after optical augmentation.

| Recipe | Affine | Optical operations |
| --- | --- | --- |
| Baseline | p=.25; +/-6 degrees; scale .95-1.05; translation +/-2% | Shared RGBIR gain p=.25, .95-1.05 |
| Enhanced | p=.35; +/-10 degrees; scale .9-1.1; translation +/-3%; 96-pixel halo | Gain p=.3, .9-1.1; gamma p=.15, .9-1.1; noise p=.1, std .005; DSM dropout p=.1 |

## Models and frozen selection

The [study config](../../configs/training/potsdam-dinov3-rgbir-v2.json) owns the
models, feature/augmentation recipes, window/stride, schedules, metrics reference
and budgets. It does not reuse DINOv2 campaign constants for those settings.

| Backbone | Revision | Blocks | Weight SHA-256 |
| --- | --- | --- | --- |
| [ViT-B/16](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) | `5931719e67bbdb9737e363e781fb0c67687896bc` | 3/6/9/12 | `9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b` |
| [ViT-L/16 satellite](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m) | `f692fa42da72c6797b67cd73494a168d1120d3ee` | 6/12/18/24 | `4e6356d992c1301b5e7c275f465e47296c5c4ad17052e262b29fc21e82ccc698` |

Both pinned snapshots passed local offline loading on a true Potsdam RGB patch.
The smoke checks verify pretrained `norm`, four register tokens, exact selected
feature tensors, frozen original parameters, finite logits and nonzero gradients
in all 48 B or 96 L LoRA B matrices. Each admitted backbone must also pass the
CUDA eight-patch >=90% train mIoU gate within 512 updates. Main runs discard fit
weights and start from pretrained, with a zero-initialized RGB residual adapter,
width-128 decoder, GroupNorm(8), stride-four detail branch and zero auxiliary
gate. The 36-patch input panel records moments, extrema, percentiles, nonfinite
counts and fractions beyond identity bounds at initialization, after small fit,
full validation and refit completion.

A uses RGB with baseline augmentation; B adds IR, NDVI, relief and availability;
C restarts the better A/B recipe with enhanced augmentation. Optional L restarts
the best A/B/C recipe using the satellite model. A/B/C/L use seed 42. D repeats
the selected recipe from pretrained with seed 43. LoRA rank/alpha/dropout are
8/16/.05 on q/k/v/o. AdamW uses 1e-4 for LoRA and 3e-4 for adapter/decoder,
5% warmup, cosine decay, gradient clipping 1 and CE/Dice .7/.3. Class weights
come only from the original 18 training scenes.

The pilot measures training, 72 fixed validation windows, one scene at 50%
overlap, horizontal-flip TTA and non-overlap diagnostics. Profiles separate
source I/O, preprocessing, host/device transfers, forward, memmap accumulation
and metric reduction. Before A, it freezes the largest common 12/8/6-pass
schedule with 25% headroom. L must fit its snapshot/VRAM/quality gates and at least
six passes; otherwise it is excluded. B failing six passes stops the campaign.
Full validation runs only at the middle and final pass, with 512 windows and
stride 256. Only those checkpoints can win.

Eligibility requires strictly greater OA than old C seed 42 and no decrease in
six-class mIoU, car IoU or clutter IoU, computed from the exact stored reference
matrix on 199,409,714 pixels. Eligible checkpoints rank by OA, mIoU, earlier
update and A/B/C/L order. Ineligible A/B parents rank by mIoU then OA. Both seed
42 and seed 43 must qualify before refit. Flip TTA is retained only if both
seeds gain >=.1 OA percentage point without reducing the other three metrics
and inference fits the budget. TTA averages logits after undoing the flip.

## Continuation, benchmark and budget

The selection lock binds configuration/source hashes, seed-42 checkpoint, seed-43
confirmation, feature statistics, inference policy and exact refit schedule.
Refit continues seed-42 weights with new AdamW/scaler, half learning rates and
one complete 24-scene pass. Its final checkpoint is update 1,152. Training loss
and mIoU in this phase are train metrics; the former validation scenes cannot
choose checkpoints, stop early or change hyperparameters.

Only a verified complete refit checkpoint can open the benchmark cache. A
single-attempt marker binds its checksum and inference policy before scoring
14 scenes and exactly 467,862,855 eroded-label pixels. Reports recompute aggregate
and per-scene OA/mIoU/class IoU/F1 from confusion matrices, remaining errors,
distance to 99%, latency and actual allocated GPU-hours. These 14 scenes were
observed in earlier campaigns, so this is a historical benchmark comparison.
It cannot isolate the benefit of adding the six refit scenes.

| Phase | Allocated GPU-hours |
| --- | ---: |
| Preflight, fit and profiles | 2 |
| Selection and seed 43 | 16 |
| Refit | 2 |
| Benchmark | 2 |
| Reserve | 2 |
| Total | 24 |

All visible allocated devices count, even when one performs model computation.
Worker execution accounting starts before dependency installation; retries retain
their prior charges. These logs are execution accounting, not provider billing
exports. An interrupted in-progress budget fails closed until its remote duration
is reconciled. A completed phase cannot be resubmitted through the normal runner.
Historical experiments are not charged against this new budget.

## Commands and artifacts

Use the project environment with train/models/vision/data-versioning extras.
Source snapshot and cache commands are CPU preparation; study phases require CUDA.

```bash
python -m deepnet.potsdam_rgbir.prepare --help
python -m deepnet.potsdam_rgbir.model --help
python -m deepnet.potsdam_rgbir.study --help
python -m scripts.run_potsdam_rgbir_kaggle --help
pytest tests/unit
```

Keep an external `RUN` directory containing `selection-cache`, `calibration` and
private Kaggle input packages. The packager embeds an allowlisted source archive;
it never uploads the worktree, credentials or old refit checkpoint for selection.
Every phase has a submission receipt, immutable input bindings, remote logs,
output hash inventory and a local audit. Recover the existing receipt after a
monitor disconnect; do not submit the phase again.

```bash
python -m scripts.run_potsdam_rgbir_kaggle package \
  --run "$RUN" --phase preflight --cache "$RUN/selection-cache"
python -m scripts.run_potsdam_rgbir_kaggle campaign --data "$DATA_ROOT" --run "$RUN"
```

The coordinator resumes already audited phases, follows the frozen roster,
stops on quality/confirmation failures, and creates role-specific private refit
and benchmark inputs only after their gates. A submission/upload with an
unconfirmed receipt requires remote reconciliation before another attempt.
Dataset/cache/checkpoint/log files stay outside Git; only DVC pointers, source,
tests, configuration and documentation are committed.
