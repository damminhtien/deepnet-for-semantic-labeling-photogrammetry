# Reproduce the figures

Render the charts from saved metrics or count tables with SciencePlots
(`science` and `no-latex`). Use the repository root and its locked notebook
environment:

```bash
uv sync --locked --extra notebooks
```

The renderer uses a CPU and needs no TeX installation, model weights, or raster
files.

## DINOv3 reports

```bash
uv run --locked --extra notebooks python -m scripts.render_scienceplots dinov3 \
  /path/to/runs/potsdam-dinov3-rgbir-v2 docs/experiments/figures
```

The selection chart reads full validation passes 4 and 8 from the three seed-42
manifests, the historical OA floor from the campaign config, and the six class
IoUs from `review-selection-abc-20261106/ABC-classes.csv`. The benchmark chart
reads `per-class.csv` and `per-scene.csv` from `review-final-20261106`. The renderer
checks these inputs against the saved audit hashes. The original review images
and their checksums remain in those directories.

## Consolidated README benchmark

```bash
uv run --locked --extra notebooks python -m scripts.render_scienceplots model-comparison \
  /path/to/runs/potsdam-pspnet-rgb-v1 \
  docs/experiments/figures/potsdam-four-model-benchmark-scienceplots-2026-11-07.png \
  --dinov3-run /path/to/runs/potsdam-dinov3-rgbir-v2
```

This figure combines the verified SAM, DINOv2 refit, DINOv3 RGB refit, and
PSPNet RGB refit results: OA and six-class mIoU, IoU by class, per-scene OA
difference from DINOv3, and recorded full-scene scoring time. This is the
complete scoring time, not model-forward-only timing. The renderer
checks the DINOv3 review inventory, PSPNet output inventory and phase audits,
checkpoint hashes and shared scene and pixel support; aggregate scores are
recomputed from confusion matrices, and per-scene scores are checked against
the audited tables and benchmark outputs. It reuses only saved metrics; it does
not run inference. The benchmark scenes were scored
in earlier campaigns, so the figure describes this historical cohort rather
than an untouched independent test set.

## EDA and legacy notebooks

```bash
uv run --locked --extra notebooks python -m scripts.render_scienceplots eda \
  /path/to/eda/20261022T063946Z /path/to/eda/scienceplots-20261022T063946Z
uv run --locked --extra notebooks python -m scripts.render_scienceplots legacy-potsdam \
  Insight-data-potsdam.ipynb /path/to/Insight-data-potsdam-scienceplots.ipynb
```

The EDA command renders seven charts to a new directory. The example CSV snapshot
uses its historical Potsdam 18/20 train/validation split and Vaihingen 12/4 split.
Use tables from one run together; its label counts, band statistics, and scene
roles are specific to that run. The EDA notebook calls the same renderers when
producing charts from its current tables.

The legacy command uses the six printed counts for each of 38 Potsdam scenes
(36,000,000 pixels per scene). It replaces the pie chart outputs while retaining
the label images and printed counts. The five legacy loss curves in `Unet.ipynb`,
`unet101_2.ipynb`, `fcn_resnet101_3sc.ipynb`, `pspnet_2.ipynb`, and
`segnet_nocyclr.ipynb` under `model_kernel/` remain as recorded. Each saved cell has
one logged loss value, too little to reproduce its curve.
