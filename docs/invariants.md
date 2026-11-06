# Data and experiment invariants

## Source assets and contracts

- Raw imagery and labels are immutable inputs. Preparation and bundle export write
  derived artifacts to separate destinations.
- Every `RasterAsset` uses a safe data-root-relative URI and declares named bands,
  dtype, dimensions, nodata, grid metadata, and an optional SHA-256.
- Band names require source evidence. The audited RGBIR source is R/G/B/IR;
  historical Potsdam three-band files match RGBIR 4/2/3 despite their configured
  IR/red/green names. Preserve those historical inputs and contracts; new RGB
  and IR/red features use independently verified bands 1/2/3 and 4/1.
- Equal dimensions alone never prove spatial alignment. Georeferenced assets need a
  matching CRS and affine transform or an explicit reprojection policy. Pixel-local
  pairs need recorded source evidence.
- Catalogs describe observed sources and do not invent missing CRS, affine, license,
  or registration metadata. Unknown source facts stay unknown.
- Read windows lazily. Do not load every 6000 by 6000 scene into RAM just to sample a
  patch.

## Labels, preprocessing, and model views

- ISPRS six-class labels use model output IDs 0–5. Black ID 6 is ignore, not a
  seventh model class. Unknown valid RGB label colors are errors.
- NoData validity is represented separately from label ignore. Invalid pixels do
  not contribute to normalization, loss, or metrics.
- Canonical preprocessing is deterministic and versioned. Model views select named
  bands and define normalization without changing source files.
- `DomainSpec` assigns each selected band an optical or auxiliary role. RGB reaches
  a pretrained encoder through a configured identity path only when channel names
  and order match. A configured residual adapter starts from channel identity
  with a zero-initialized 1x1 residual; legacy projection behavior stays bound to
  its original model spec.
- Auxiliary channels use a separate spatial branch and do not receive optical
  brightness or contrast transforms.
- Geometric augmentation runs on raw, aligned channels, labels, and validity masks
  before deterministic model-view normalization. Per-sample seeds make the result
  independent of DataLoader worker scheduling.
- Any fitted statistics use only the train scenes and record the exact split hash
  and scene IDs. A changed split or fitted-statistics result creates a different
  preprocessing contract hash.
- Stochastic augmentation is restricted to training. Validation and test sampling
  use deterministic windows with full valid-pixel coverage.
- Verified caches preserve source arrays, masks, class histograms and scored-pixel
  counts. Their hashes include source, split, preprocessing, feature recipe and
  label roles. Selection, refit and benchmark caches have separate scopes.
- DSM availability is independent of image/label validity. Missing DSM sets relief
  and availability to zero without removing scored pixels, including the missing
  column in Potsdam `3_13`. JPEG nDSM remains blocked. Relief statistics stay fitted
  on the original 18 training scenes when refit adds the former validation scenes.

## Splits and leakage

- Split assignments happen at scene or declared spatial-group level before patch
  generation. Scene and spatial-group intersections across splits are invalid.
- `potsdam-legacy-v1` preserves 18/6/14 scene membership from older experiments.
  It is scene-level and not region-held-out. Scores are bound to each experiment's
  original data, preprocessing and inference contracts. The completed DINOv3
  campaign compares systems on the historically observed 14-scene benchmark.
- Evaluation APIs accept scene IDs only from the selected validation or test
  manifest. Class-aware sample indexes can use training labels only.
- Do not substitute withheld benchmark labels or change a frozen split to work
  around missing source data.

## Selection, continuation and final scoring

- Potsdam DINOv3 selection uses only 18 train and six validation scenes. Main arms
  start from pinned pretrained weights; small-fit weights are discarded. Only
  scheduled full-scene validation checkpoints can win, and seed 43 must satisfy
  the same four metric floors before the selection is locked.
- Refit derives a separate train-only split from the frozen 18+6 union, with
  participant labels for all 24 scenes and no eroded or benchmark labels. It loads
  the winning seed-42 weights, resets AdamW/scaler and runs exactly 1,152 updates.
  Metrics on those 24 scenes are training metrics; they cannot select checkpoints
  or change the locked recipe, feature statistics or inference policy.
- The final checkpoint hash and completed refit audit gate one benchmark attempt.
  The DINOv3 protocol scores exactly 467,862,855 pixels with 512x512 windows and
  stride 256. Missing auxiliary data cannot reduce that denominator.
- Saved aggregate and per-scene confusion matrices are the source for metric
  recomputation. A completed benchmark is reviewed from those artifacts without
  rerunning inference or tuning against its labels. The result is a historical
  local benchmark comparison, not a fresh independent test or an official ISPRS
  leaderboard submission.

## Runtime and provenance

- Local and Kaggle consume the same portable bundle. Only data root, execution
  device, loader worker count, and output root vary by host.
- CUDA is selected when available, then Apple MPS, then CPU. CUDA-only transfer
  options are disabled on local CPU/MPS runs.
- A run manifest records bundle, catalog, split, preprocessing, model-input,
  sampler, source revision, configuration, seed, and runtime metadata.
- A successful static check or test does not establish model training completion or
  reproduce a published score. Training, validation, and held-out results remain
  separate claims.
- Model output dimensions are defined by each task spec. The ignore ID is excluded
  from output classes and must be honored by the caller's loss and metrics.
- The DINOv3 license reference and restrictions travel with its model spec. Check the
  selected weights' license before use; choosing another backbone does not change
  data contracts or decoder interfaces.

## Historical experiment records

Reports and notebook outputs retain the contracts and evidence of their original
experiments. Preserve them unchanged and distinguish completed, independently
audited campaigns from unexecuted configurations. Comparisons across recipes do
not isolate the effect of one backbone or feature. The old PSPNet/SAM training
runners and `semantic_labeling` data helpers remain historical source; they are
not verified consumers of the new bundle format. Build new runs from checked
`deepnet.data` bundles or the verified role-scoped DINOv3 caches.
