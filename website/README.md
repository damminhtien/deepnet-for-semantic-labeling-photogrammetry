# DeepNet Research Atlas

An Astro site for exploring the DeepNet research story, spectral identity,
sampling, augmentation, model tensors, saved experiments and Potsdam evaluation.

From the repository root, with Node 24 and Python 3:

```bash
npm --prefix website ci
npm --prefix website run dev
```

Open the URL printed by Astro at
`/deepnet-for-semantic-labeling-photogrammetry/`. Use
`npm --prefix website test` and `npm --prefix website run build` to validate a
change. The static build uses an explicit allowlist of research metadata and
committed visual assets. It checks the five archived figures and all 20 curated
raster crops. Each crop contains lossless RGB and infrared PNGs plus an RGB
class-ID plane for reference, DINOv3, and PSPNet. Validation checks asset
checksums, dimensions, class IDs, checkpoint bindings, and each crop's exact
three-way confusion counts. The build does not need the original scene archive,
local prediction outputs, a GPU, or a backend. The Pipeline starts with one
recorded refit training draw, replayed through the implemented sampler and
augmentation function, the final DINOv3 checkpoint, real block activations, and
sample-only error counts. Its separate Explain controls remain labelled
schematics.

The selected excerpts use project-owner-authorized publication of rendered
imagery and labels. The Atlas publishes a bounded set of visual assets, not a
mirror of the full ISPRS Potsdam archive. Generated manifests record the
authorization and link the [ISPRS Potsdam dataset
page](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)
for provenance. The public build requires only committed assets and metadata.

The route controls encode choices in the URL. The Data Atlas also accepts a
local optical TIFF. The Potsdam viewer checks the file against the selected
scene's audited RGBIR SHA-256; the Vaihingen viewer checks band count and bit
depth and leaves file identity unverified. The spectral histogram describes the
training collection, separate from the source TIFF. Its committed public sample
also supports pixel inspection and a swipe between the observed historical
B4/B2/B3 mapping and the B4/B1/B2 mapping implied by historical names. Both
composites preserve the recorded digital numbers and only change channel order;
lower overlay opacity blends the display composites.

## Refresh the committed public crops

The exporter uses the locked held-out cache and the indexed paired prediction
run. It verifies source TIFF hashes, raw RGBIR values, decoded eroded labels,
prediction hashes, model checkpoints, and all crop transition counts before
writing any assets:

```bash
.venv/bin/python website/scripts/export_public_crops.py \
  --cache /Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/runs/potsdam-dinov3-rgbir-v2/heldout-cache \
  --paired-output data/runs/potsdam-paired-error-study-v1/attempt-3/outputs \
  --runs-root /Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/runs
```

The output is the bounded bundle under `website/public/evidence/scenes/` and
`website/src/generated/public-crops.json`. A clean clone can run the static
build and validate those assets without the local cache and prediction files.

## Refresh the recorded training trace

The offline exporter checks the refit cache, protocol, selection lock, pinned
pretrained snapshot, checkpoint bytes, and checkpoint binding before it draws
one deterministic first-pass training window and runs one CPU inference. It
exports rendered PNGs and provenance metadata only:

```bash
.venv/bin/python website/scripts/export_pipeline_trace.py \
  --runs-root /path/to/runs \
  --pretrained /path/to/dinov3-vitb16-pretrain-lvd1689m \
  --device cpu
```

The default output is `website/public/evidence/pipeline-trace/` plus
`website/src/generated/pipeline-trace.json`. The checkpoint and cache remain
outside the site. The example comes from final refit training data; its
prediction and confusion counts are explicitly in-sample, not benchmark
measurements. Activation PNGs are PCA projections of the real block 3, 6, 9,
and 12 feature tensors. They are not attention maps or comparable activation
magnitudes.

## Inspect paired prediction rasters privately

With the repository's Python environment, the archived paired inference output,
the original eroded-label test cache and the two run archives:

```bash
python website/scripts/export_local_crops.py \
  --kaggle-output /path/to/paired-run/outputs \
  --cache /path/to/heldout-cache \
  --runs-root /path/to/runs \
  --output /path/outside-the-repo/new-crop-directory
```

Use the homepage, Data Atlas, Pipeline, or Evaluation Observatory to inspect a
committed pixel-aligned example without uploading anything. Use **Evaluation →
Swipe through a real crop** to inspect any larger local crop package.
The exporter checks cache and prediction hashes, split, label protocol, checkpoint
and scene geometry. It also reproduces each published crop's full joint class
counts before writing its three aligned class-ID planes. The browser checks
the package metadata and recomputes those counts again; a histogram match alone
does not independently authenticate an arbitrary file's spatial arrangement.
Treat crops and downloaded comparison images as local research data.

## Publish on GitHub Pages

Set **Settings → Pages → Build and deployment → Source** to **GitHub Actions**.
Once the commit is on `master`, run **Actions → Research Atlas → Run workflow**
on `master`. Pushes and PRs build and validate the site; publication is manual.
The expected project URL is
[DeepNet Research Atlas](https://damminhtien.github.io/deepnet-for-semantic-labeling-photogrammetry/).

The build includes the SIL Open Font License notices for the three font packages. For dataset evidence
and distribution scope, see [the public evidence bundle](../artifacts/public/README.md).
