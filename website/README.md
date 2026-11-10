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
change. The build runs deterministic Python exporters over an explicit allowlist
of research metadata and five paired-case figures already tracked under
`docs/research/figures/`. The visual exporter copies those PNGs byte-for-byte,
records their checksums and panel geometry, then validates case coordinates,
class counts and checkpoint bindings. It never reads private rasters or
prediction files. Public comparison panels are rendered portions of those
archived figures (411–412 × 411 pixels), not the original 512 × 512 rasters or
class-ID planes. The pipeline sampling and augmentation controls remain
educational schematics and say so in the interface.

The tracked figures contain derived ISPRS imagery and labels. The repository
does not grant rights to redistribute the source dataset or pretrained weights;
the generated manifest records that status. Review the provider's terms before
mirroring these figures outside this repository. The [ISPRS Potsdam dataset
page](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)
describes the imagery and bands; it is linked for provenance, not as a grant of
reuse rights. The public build does not
require a model, GPU, external map service or database.

The route controls encode choices in the URL. The Data Atlas also accepts a
local optical TIFF. The Potsdam viewer checks the file against the selected
scene's audited RGBIR SHA-256; the Vaihingen viewer checks band count and bit
depth and leaves file identity unverified. The spectral histogram describes the
training collection, separate from the source TIFF.

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

Use **Evaluation → Follow the selected error into a scene** to inspect an
archived paired-case figure. Use **Evaluation → Swipe through a real crop** to
open one `.atlas.json` file exported from the private paired rasters.
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
