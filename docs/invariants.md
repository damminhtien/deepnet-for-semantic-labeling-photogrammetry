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
- Georeferenced assets need a matching CRS and affine transform or an explicit
  reprojection policy. Pixel-local pairs need recorded source evidence.
- Record only verified CRS, affine, license and registration metadata.
- Read windows lazily.

## Labels, preprocessing, and model views

- ISPRS outputs use IDs 0–5 and ignore black label ID 6. Reject unknown valid RGB
  label colors.
- Track NoData separately from label ignore. Use valid pixels for normalization,
  loss and metrics.
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
- `potsdam-legacy-v1` preserves the 18/6/14 scene split from older experiments.
  Neighboring scenes can belong to different splits. Report scores with the
  original data, preprocessing and inference contracts.
- Evaluation APIs accept scene IDs only from the selected validation or test
  manifest. Class-aware sample indexes can use training labels only.
- Use the labels and scenes declared in the frozen split.

## Selection, continuation and final scoring

- Potsdam DINOv3 selection uses only 18 train and six validation scenes. Main arms
  start from pinned pretrained weights; small-fit weights are discarded. Only
  scheduled full-scene validation checkpoints can win, and seed 43 must satisfy
  the same four metric floors before the selection is locked.
- Refit uses a separate train-only split with participant labels for all 24 scenes
  in the frozen 18+6 union. It loads the winning seed-42 weights, resets AdamW and
  the scaler, and runs 1,152 updates. Keep the recipe, feature statistics and
  inference policy fixed. Report metrics on those 24 scenes as training metrics.
- The final checkpoint hash and completed refit audit gate one benchmark attempt.
  The DINOv3 protocol scores exactly 467,862,855 pixels with 512x512 windows and
  stride 256. Use the image and label masks to count scored pixels.
- Recompute benchmark metrics from saved aggregate and per-scene confusion
  matrices. Report them as local scores on the historical benchmark.

## Runtime and provenance

- Local and Kaggle consume the same portable bundle. Only data root, execution
  device, loader worker count, and output root vary by host.
- CUDA is selected when available, then Apple MPS, then CPU. CUDA-only transfer
  options are disabled on local CPU/MPS runs.
- A run manifest records bundle, catalog, split, preprocessing, model-input,
  sampler, source revision, configuration, seed, and runtime metadata.
- Report training, validation and held-out metrics separately.
- Model output dimensions are defined by each task spec. The ignore ID is excluded
  from output classes and must be honored by the caller's loss and metrics.
- Check the license for the selected DINOv3 weights in the model spec before use.

## Historical experiment records

Keep experiment contracts, source artifacts and saved outputs from each run.
PSPNet and SAM use the legacy `semantic_labeling` helpers and their original
bundle formats. New runs use `deepnet.data` bundles or the verified DINOv3
caches.
