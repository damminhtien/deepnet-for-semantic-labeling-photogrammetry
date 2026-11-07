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
