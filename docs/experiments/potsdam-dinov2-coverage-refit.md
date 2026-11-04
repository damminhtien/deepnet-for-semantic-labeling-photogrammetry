# Potsdam DINOv2: selection, refit, held-out evaluation

This study tests a smaller normalized ViT-B/14 decoder and a channel-preserving
IRRG adapter. A/B/C select features and augmentation on the existing 18/6 split.
A seed-43 confirmation must pass the same validation floors. Only then may the
winner be continued on 24 participant-labeled scenes. The 14 test scenes are
scored once, using the final refit checkpoint. OA 99% is the target.

Use the project environment with the model, training and vision extras.
Keep the study directory, bundles, checkpoints and logs outside Git.
The DINOv2 snapshot must contain its audited pretrained manifest.

## Inputs and pilot

Set `STUDY` to a new external directory, `SELECT_BUNDLE` to the verified Potsdam
train/validation bundle, `DINO2` and `DINO3` to their local snapshots, and
`HISTORICAL` to the audited DINOv2 joint-run best checkpoint.

```sh
python -m deepnet.potsdam_study smoke-dinov3 \
  --root "$STUDY" --bundle "$SELECT_BUNDLE" --pretrained "$DINO3" --device cpu

python -m deepnet.potsdam_study preflight \
  --root "$STUDY" --bundle "$SELECT_BUNDLE" --pretrained "$DINO2" \
  --old-checkpoint "$HISTORICAL" --dinov3-smoke "$STUDY/dinov3-smoke.json" \
  --dsm "$DSM_FEATURES"
```

DSM is optional at pilot time. Omit `--dsm` to lock the spectral-only B if DSM
provenance cannot be verified. To regenerate DSM features, use
`dvc checkout data/auxiliary/potsdam-dsm-v1.dvc` with the matching cache, then:

```sh
python -m deepnet.potsdam_study prepare-dsm \
  --catalog data/prepared/potsdam/catalog.json \
  --raw data/auxiliary/potsdam-dsm-v1/1_DSM \
  --inventory data/auxiliary/potsdam-dsm-v1/inventory.json \
  --output "$DSM_FEATURES"
```

The DSM inventory was obtained from `Potsdam/1_DSM.rar` inside the source
`Potsdam.zip`; all 38 extracted TIFF hashes matched the pre-existing rasters.
Only DSM and its world files are in this DVC snapshot. The feature manifest
records the raw hashes, grids, extent, coverage and train-only scale. DSM
availability is a feature; it never removes a pixel from scoring. The JPEG
nDSM remains excluded. Kaggle input datasets must remain private.

The pilot writes the 36-patch backbone distribution before training, after
the eight-patch fit and for the historical adapter. It also measures nine
training updates, an independent 72-window validation probe and one
validation scene at both stride 252 and 504. Only the stride-252 route is
used for scoring. The runtime report separates raster reads, transfers,
forward, accumulation and metric reduction. The pilot freezes four or six
passes only after passing the fit and budget gates.

## Selection, continuation and test

Run `select --arm A`, `B`, `C`, then `D` with the same
`--root`, `--bundle`, `--pretrained`, device and applicable DSM arguments.
C starts from fresh model weights with the recipe selected between A/B; D
repeats the winner at seed 43. Each selection pass reads 2,592 base windows
and adds 432 windows for each rare class. Full validation at even passes
selects among checkpoints meeting the SAM OA, mIoU, car and clutter floors.
The patch probe does not affect ranking. The immutable source hashes,
matrices and checkpoints are recorded in the study output.

```sh
python -m deepnet.potsdam_study pack-refit \
  --root "$STUDY" --bundle "$SELECT_BUNDLE" --raw "$ISPRS_INPUT_ROOT" \
  --output "$REFIT_BUNDLE"

python -m deepnet.potsdam_study refit \
  --root "$STUDY" --bundle "$REFIT_BUNDLE" --pretrained "$DINO2" \
  --dsm "$DSM_FEATURES"

python -m deepnet.potsdam_study heldout \
  --root "$STUDY" --bundle "$HELDOUT_BUNDLE" --pretrained "$DINO2" \
  --dsm "$DSM_FEATURES"
```

`pack-refit` requires the validation winner and seed-43 confirmation. It
exports participant labels for all 24 scenes. Point `ISPRS_INPUT_ROOT` at the
verified original input tree containing those 24 participant labels. The
default DVC train/validation staging contains participant labels only for the
18 selection-training scenes, so it cannot supply the other six.
The refit job checks original IRRG and training label hashes against selection,
inherits the winner's
weights, feature scale, class weights and augmentation, and resets AdamW and
AMP state at half learning rate. It trains one 1,152-update pass, without
running validation. A paused training stage can resume from its bound
`last.pt` with the original phase budget file. The held-out command requires
`final.pt` at the completed refit step and records a single attempt before
opening the test bundle. Any attempted held-out run consumes that attempt.
Do not retry held-out against a different checkpoint.

## Kaggle packaging

`python -m scripts.package_potsdam_kaggle inputs --help` stages the calibration
input and the optional DSM inputs in separate private datasets. Upload the
selection DSM input with the pilot; hold the test DSM input for evaluation.
Use `python -m scripts.package_potsdam_kaggle kernel PHASE --output DIR` for
`preflight`, `A`, `B`, `C`, `D`, `refit` and `heldout`. Add
`--with-dsm` when DSM was locked in the pilot and `--heldout-dsm` for
held-out only if the winner uses it. The source payload is hashed; the
stage mounts only the data role it can read. Use the packager's
`refit-input --bundle "$REFIT_BUNDLE" --output DIR` and upload it with
`kaggle datasets create -p DIR -r tar` after locking selection.
The default Kaggle package expects two T4 GPUs. Submit it with
`--accelerator NvidiaTeslaT4` and timeouts of **1,800 seconds for the pilot,
8,100 per selection run, and 3,600 each for refit and held-out**. The runner
uses one GPU but charges the budget for both allocated GPUs. For a verified
single-GPU allocation, package with `--allocated-devices 1` and double those
wall-time limits. The phase caps total 23 GPU-hours; the remaining GPU-hour
covers setup failures or other reserve use. Submit downstream jobs only
after verifying their prerequisites.

Report validation metrics for the pre-refit checkpoint. Report held-out
metrics and the OA shortfall only from a complete run on 467,862,855
pixels. This design does not measure the isolated held-out effect of
adding the six scenes.
