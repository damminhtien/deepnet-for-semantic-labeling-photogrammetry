# Frozen encoder segmentation probes

> **Historical results — rerun required.** These experiments predate the
> contract-driven data pipeline. Their metrics remain archived evidence and have
> not been reproduced through the current catalog, preprocessing, or evaluation
> contracts.

Date: 2026-10-19
Environment: Python 3.14.8, PyTorch 2.14.1, torchvision 0.29.1, Apple MPS.

These are small supervised linear-probe experiments. The pretrained encoders stay frozen; only one linear pixel classifier is trained for 100 epochs. Results are coarse-grid validation measurements, not full-resolution benchmark submissions.

## Potsdam validation

The fixed split uses 18 labeled scenes for training and 6 scenes for validation. The 14 held-out scenes were not accessed in this pilot. Images use IR-R-G remapped to R-G-NIR-as-blue; DSM and nDSM are omitted. The feature grid is compared at approximately 5 m and 10 m per cell.

### Approximately 5 m grid, six classes

| Encoder | Validation OA | Validation mIoU | Car IoU | Feature grid |
|---|---:|---:|---:|---:|
| UniverSat | 52.07% | 23.75% | 1.57% | 60 x 60 |
| AnySat | 41.24% | 17.15% | 0.53% | 30 x 30, resized to 60 x 60 |
| TerraMind | 46.07% | 20.39% | 0.92% | 14 x 14, resized to 60 x 60 |

The majority-class baseline is 30.39% OA and 5.07% mIoU. Cars have only 113 training cells and 57 validation cells after reduction, so their IoU has very small support.

### Approximately 10 m grid, five represented classes

| Encoder | Validation OA | mIoU over represented classes |
|---|---:|---:|
| UniverSat | 58.74% | 33.16% |
| AnySat | 53.46% | 27.71% |
| TerraMind | 50.28% | 27.10% |

Majority reduction at this scale removes cars from both training and validation. These are five-class scores and must not be compared with the six-class results above.

## RGB dataset probes

Each model uses a frozen encoder and the same 96 train plus 96 validation samples per dataset. Outputs and labels are reduced to a 32 x 32 grid. The scores below use only the sampled validation data.

| Dataset | Encoder | Validation OA | Validation mIoU | Majority baseline mIoU | Evaluated cells |
|---|---|---:|---:|---:|---:|
| LandCover.ai v1 | UniverSat | 77.36% | 42.07% | 10.41% | 98,304 |
| LandCover.ai v1 | AnySat | 57.79% | 23.80% | 10.41% | 98,304 |
| LandCover.ai v1 | TerraMind | 79.82% | 45.24% | 10.41% | 98,304 |
| LoveDA v1 | UniverSat | 40.43% | 23.00% | 5.22% | 96,322 |
| LoveDA v1 | AnySat | 20.13% | 11.67% | 5.22% | 96,322 |
| LoveDA v1 | TerraMind | 48.01% | 28.33% | 5.22% | 96,322 |

TerraMind has the highest sampled mIoU on both RGB datasets. This is preliminary evidence for further evaluation, not enough to select a production model.

### Per-class IoU

| LandCover.ai class | UniverSat | AnySat | TerraMind |
|---|---:|---:|---:|
| Background | 68.13% | 52.52% | 69.98% |
| Building | 20.95% | 16.23% | 20.31% |
| Woodland | 65.35% | 33.56% | 73.31% |
| Water | 42.31% | 11.59% | 45.79% |
| Road | 13.60% | 5.08% | 16.79% |

| LoveDA class | UniverSat | AnySat | TerraMind |
|---|---:|---:|---:|
| Background | 26.68% | 7.24% | 37.04% |
| Building | 34.33% | 23.32% | 31.46% |
| Road | 19.16% | 9.56% | 40.22% |
| Water | 40.20% | 8.50% | 32.72% |
| Barren | 4.35% | 8.63% | 5.94% |
| Forest | 11.76% | 6.51% | 24.33% |
| Agriculture | 24.55% | 17.96% | 26.65% |

## Procedure and limits

- The linear head uses AdamW, learning rate 0.02, weight decay 0.0001, square-root inverse-frequency class weights from training targets, and seed 20261019.
- Normalization statistics are estimated from selected training imagery only. Potsdam uses its IR-R-G imagery; the external datasets use RGB. Labels are reduced by majority class, and their ignore IDs are omitted from scoring.
- For LandCover.ai, validation orthophotos are disjoint from training orthophotos. The probe samples 96 non-overlapping 256 m patches in each split.
- For LoveDA, 96 train and 96 validation images are sampled across urban and rural domains. Only the public Train and Val archives were evaluated; no test archive was accessed.
- Coarse grids lose boundaries and small objects. LandCover.ai building/road support is sparse in the selected validation patches. Scores are not comparable with published full-resolution results, Potsdam leaderboard scores, or one another across different label taxonomies.
- A TerraMind 480 px Potsdam attempt is excluded: a checkpoint shape mismatch caused its positional embedding to be randomly initialized. The included TerraMind probe uses the native 224 px checkpoint configuration.
- LoveDA and its source imagery have non-commercial use restrictions. Dataset archives, source images, masks, feature caches, probe weights, and previews remain outside this repository.

## Artifacts

- Aggregated metrics and provenance: [frozen-encoder-probes-2026-10-19-metrics.json](frozen-encoder-probes-2026-10-19-metrics.json).
- Deeper Potsdam/Vaihingen evaluation: [potsdam-vaihingen-foundation-probe-2026-10-19.md](potsdam-vaihingen-foundation-probe-2026-10-19.md) and its metrics JSON.
- External dataset sources: [LandCover.ai](https://landcover.ai.linuxpolska.com/) and [LoveDA v1](https://doi.org/10.5281/zenodo.5706578).
