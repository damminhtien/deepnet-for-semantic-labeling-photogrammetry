# Potsdam and Vaihingen foundation encoder evaluation

> **Historical results — rerun required.** These experiments predate the
> contract-driven data pipeline. Their metrics remain archived evidence and have
> not been reproduced through the current catalog, preprocessing, or evaluation
> contracts.

Date: 2026-10-19
Environment: Python 3.14.8, PyTorch 2.14.1, Apple MPS.

This deeper evaluation compares three frozen pretrained encoders with one supervised linear segmentation head. It is not end-to-end fine-tuning and is not an ISPRS leaderboard submission. Scores use a coarse grid of approximately 5 m per cell; label ID 6 is ignored.

## Primary results

The primary metrics use eroded/no-boundary references. Potsdam uses a fixed 18/6/14 scene split. Vaihingen uses 4-fold out-of-fold predictions over the 16 official participant-labeled tiles, then refits on all 16 and evaluates once on the 17 official held-out tiles.

| Encoder | Potsdam held-out OA | Potsdam held-out mIoU | Vaihingen 4-fold OOF mIoU | Vaihingen held-out OA | Vaihingen held-out mIoU | Potsdam to Vaihingen mIoU | Vaihingen to Potsdam mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| UniverSat | 47.22% | 22.96% | 27.69% | 58.57% | 29.51% | 10.11% | 9.41% |
| AnySat | 31.57% | 13.73% | 21.99% | 56.27% | 27.86% | 11.23% | 11.91% |
| TerraMind | 48.04% | 22.28% | 21.18% | 50.87% | 23.56% | 9.20% | 11.59% |

The Potsdam 5 m validation probe preceding this run measured mIoU of 23.75% for UniverSat, 17.15% for AnySat, and 20.39% for TerraMind. A train-majority baseline measured 5.57% mIoU on the held-out Potsdam grid and 4.79% on the Vaihingen held-out grid.

### Reference-mask sensitivity

| Encoder | Potsdam full reference OA / mIoU | Potsdam eroded reference OA / mIoU | Vaihingen full reference OA / mIoU | Vaihingen eroded reference OA / mIoU |
|---|---:|---:|---:|---:|
| UniverSat | 47.18% / 22.96% | 47.22% / 22.96% | 58.52% / 29.45% | 58.57% / 29.51% |
| AnySat | 31.51% / 13.76% | 31.57% / 13.73% | 56.20% / 27.82% | 56.27% / 27.86% |
| TerraMind | 48.01% / 22.24% | 48.04% / 22.28% | 50.90% / 23.58% | 50.87% / 23.56% |

### Held-out per-class IoU

| Potsdam class | UniverSat | AnySat | TerraMind |
|---|---:|---:|---:|
| Impervious surfaces | 34.52% | 16.52% | 40.36% |
| Buildings | 40.24% | 27.94% | 35.78% |
| Low vegetation | 29.33% | 17.53% | 28.88% |
| Trees | 22.22% | 15.31% | 19.19% |
| Cars | 1.16% | 0.60% | 0.88% |
| Clutter/background | 10.31% | 4.45% | 8.58% |

| Vaihingen class | UniverSat | AnySat | TerraMind |
|---|---:|---:|---:|
| Impervious surfaces | 45.67% | 45.53% | 40.50% |
| Buildings | 44.80% | 44.35% | 35.76% |
| Low vegetation | 31.75% | 27.22% | 23.70% |
| Trees | 43.99% | 39.51% | 36.32% |
| Cars | 2.35% | 1.52% | 2.21% |
| Clutter/background | 8.49% | 9.03% | 2.84% |

Cars score poorly for every encoder at this grid size. Majority reduction to approximately 5 m removes much of their small-object detail, so this result does not establish full-resolution car performance.

## Protocol

- The Potsdam split uses 18 participant-labeled training scenes, 6 validation scenes, and 14 held-out scenes. Held-out labels were used only for this final evaluation.
- Vaihingen uses 16 official participant-labeled scenes for supervised probes and the 17 remaining official tiles for one held-out evaluation. The 4-fold shuffled split is scene-level; reported OOF predictions cover the 16 training tiles. Participant labels match the complete reference on those training tiles.
- Both datasets use IR-R-G imagery remapped to R-G-NIR-as-blue. DSM and nDSM are omitted. Mean and standard deviation are estimated from source training imagery only.
- Targets are majority-class reduced to a common approximately 5 m grid. UniverSat emits 60 x 60 features; AnySat emits 30 x 30 and TerraMind 14 x 14, which are bilinearly resized to 60 x 60. The different native feature resolutions limit direct comparison.
- Each probe trains for 100 epochs with AdamW, learning rate 0.02, weight decay 0.0001, square-root inverse-frequency class weighting from training labels, and seed 20261019.
- Cross-domain scores apply the source-trained head and source normalization directly to the other dataset without target-label use or target-head adaptation. The low scores indicate substantial domain shift.

## Interpretation

UniverSat has the strongest Vaihingen OOF and held-out mIoU and the strongest Vaihingen per-class scores overall. TerraMind has the highest Potsdam held-out OA and impervious-surface IoU, while UniverSat has slightly higher Potsdam mIoU. None of the encoders dominates every dataset and metric.

The measured cross-domain mIoUs are all below 12%. The RGB probes in [frozen-encoder-probes-2026-10-19.md](frozen-encoder-probes-2026-10-19.md) favor TerraMind on sampled LandCover.ai and LoveDA data, while this deeper urban benchmark favors UniverSat on Vaihingen. These are reasons to keep both in the candidate set; they do not settle a single flagship choice for a multi-source online pipeline.

Evaluation elapsed time after model/data preload was 124.1 s for UniverSat, 118.7 s for AnySat, and 112.2 s for TerraMind. Each time includes four-fold OOF probes, final in-domain fits/evaluations, and both cross-domain evaluations. These are not per-tile inference latencies and must not be used as an online throughput comparison. TerraTorch emitted a Python 3.14 `torch.jit.script` warning; TerraMind inference completed.

## Data and provenance

Vaihingen imagery and reference masks came from the [official ISPRS benchmark page](https://isprs.org/resources/datasets/benchmarks/UrbanSemLab/default.aspx). The download fetched only selected 2D imagery and labels from the linked archive; DSM, ALS/LAS, 3D semantic-labeling, and reconstruction members were excluded. Potsdam data were already available locally. Raw images, masks, model weights, trained heads, feature caches, logs, predictions, and preview images remain outside Git.

The machine-readable metrics JSON records train/validation/held-out scene IDs, folds, per-scene metrics, confusion counts, source revisions, checkpoint hashes, archive checksums, and evaluation-script hashes: [potsdam-vaihingen-foundation-probe-2026-10-19-metrics.json](potsdam-vaihingen-foundation-probe-2026-10-19-metrics.json).
