# Data and experiment invariants

## Source assets and contracts

- Raw imagery and labels are immutable inputs. Preparation and bundle export write
  derived artifacts to separate destinations.
- Every `RasterAsset` uses a safe data-root-relative URI and declares named bands,
  dtype, dimensions, nodata, grid metadata, and an optional SHA-256.
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
  and order match; other optical combinations use a learned 1x1 projection.
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

## Splits and leakage

- Split assignments happen at scene or declared spatial-group level before patch
  generation. Scene and spatial-group intersections across splits are invalid.
- `potsdam-legacy-v1` preserves 18/6/14 scene membership from older experiments.
  It is scene-level and not region-held-out. Existing scores are historical and need
  to be rerun against the new catalog, preprocessing, and evaluation contracts.
- Evaluation APIs accept scene IDs only from the selected validation or test
  manifest. Class-aware sample indexes can use training labels only.
- Do not substitute withheld benchmark labels or change a frozen split to work
  around missing source data.

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

The existing reports and notebook outputs describe experiments completed before the
contract-driven pipeline. Preserve them as historical evidence. Rerun them using
the new pipeline before treating any metric as current or comparing the data access
to those results. The old PSPNet/SAM training runners and `semantic_labeling` data
helpers remain historical source; they are not verified consumers of the new bundle
format. Build new runs from `deepnet.data` and a checked `BundleData` object.
