# Segmentation model and sidewalk research notes

> **Historical research snapshot.** Any experiment metrics referenced here predate
> the contract-driven data pipeline and must be rerun before they are treated as
> current results.

Date checked: 2026-10-19.

This note records the public benchmark context, the current model recommendation based on our measured probes, the SAM experiment boundary, and the best supported direction for pedestrian-path segmentation.

## Potsdam leaderboard context

The official ISPRS Potsdam 2D table currently labels the Southwest Jiaotong University submission `SWJ`; I did not find a `SWJ_2` row in that table. Its `Overall` score is 89.5%. The page states that its class-specific quality measures are F1 scores using eroded reference boundaries. The `Overall` value should therefore be cited as the benchmark's Overall score, not as mIoU.

Several official submissions list higher Overall scores, including `AMA_1` at 91.2% and `HUSTW3`/`HUSTW4` at 91.6%. The linked AMA method note describes a five-fold RefineNet ensemble. The linked HUST note is for its Vaihingen entry and describes a supervised deconvolution/U-Net approach with IRRG, DSM and nDSM plus label voting; it should not be treated as a detailed method description for each Potsdam variant. These are historical benchmark entries with their own data and scoring procedure, not directly comparable to the coarse-grid probes in this repository.

Sources: [official Potsdam results table](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/results/label_challenge_2D_potsdam_result_ifr.html), [SWJ per-tile details](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/results/2D_labeling_potsdam/2D_labeling_Potsdam_details_SWJ/index.html), [AMA_1 method description](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/results/papers/Method_Description_AMA.pdf), and [HUST brief method description](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/results/papers/HUST_WH_brief_descr.pdf).

## Provisional flagship candidate

**Recommendation for the next prototype: use TerraMind as the provisional flagship and keep UniverSat as the primary challenger.** In our sampled RGB probes TerraMind has the highest mIoU on both LandCover.ai and LoveDA, and it has the highest Potsdam held-out OA in the deeper evaluation. UniverSat has the strongest Potsdam held-out mIoU by a small margin and is clearly strongest on Vaihingen held-out mIoU. The domain-transfer results are weak for every encoder.

This recommendation is limited to the frozen-encoder linear probes that were actually run. Those probes score at about 5 m or 8-10 m per cell, and their elapsed evaluation time includes feature extraction, repeated fits and multiple evaluations. We have not measured full-resolution per-tile latency, memory under production tile sizes, or quality after end-to-end fine-tuning. The evidence supports a short list for the next online pipeline prototype; it does not establish a universal winner. See [frozen-encoder probes](../experiments/frozen-encoder-probes-2026-10-19.md) and [Potsdam/Vaihingen deep evaluation](../experiments/potsdam-vaihingen-foundation-probe-2026-10-19.md).

## SAM status

Meta's original SAM is a promptable object-mask model. Our repository adapts the SAM ViT-B image encoder with rank-4 LoRA and a new six-class semantic decoder; this is a separate supervised semantic-segmentation experiment, not zero-shot SAM evaluation. The implementation and private Kaggle job were prepared, but no completed training metrics were verified in the state recorded for this report. Do not quote a SAM accuracy until the CUDA run and held-out evaluation complete.

Sources: [official Segment Anything repository](https://github.com/facebookresearch/segment-anything) and [SAM paper](https://arxiv.org/abs/2304.02643).

## Pedestrian paths and sidewalks

**Tile2Net is the strongest direct candidate found for high-resolution aerial pedestrian infrastructure.** Its current repository detects roads, sidewalks, crosswalks and footpaths, then produces polygons and a pedestrian network. Its published 2023 paper reports 82.67% sidewalk IoU, 75.42% crosswalk IoU and 84.51% four-class mIoU in that paper's evaluation. The current repository says its beta release has an updated model, so those paper scores do not describe today's model weights.

Tile2Net currently requires a CUDA-enabled GPU for inference. Its guide prefers orthorectified imagery at zoom 19 or higher, says not to use less than zoom 18, and cautions that a different built environment may require additional training. Covered paths, canopy, shadows, and resolution differences remain practical concerns for transfer to new cities. The software supports Python 3.11 through 3.14, but CUDA hardware is still required; this machine's Apple MPS setup is not the documented inference target.

The [APE dataset](https://sidewalks.washington.edu/2024/03/01/ape/) is a useful transfer-learning/evaluation source. It provides co-registered aerial imagery, street-map tiles and human-validated masks for sidewalks, crossings and corner bulbs across selected regions in the Americas. It offers labels that Potsdam and Vaihingen do not: those ISPRS taxonomies combine roads and sidewalks under impervious surfaces and have no separate sidewalk class.

Sources: [Tile2Net repository and current runtime requirements](https://github.com/VIDA-NYU/tile2net), [Tile2Net project preparation guide](https://github.com/VIDA-NYU/tile2net/blob/main/BASICS.md), [2023 Tile2Net paper](https://fmiranda.me/publications/pdfs/tile2net.pdf), and [University of Washington APE dataset description](https://sidewalks.washington.edu/2024/03/01/ape/).

## Next evaluation that would resolve the remaining choice

1. Run TerraMind and UniverSat on the same full-resolution, held-out aerial scenes with tiled inference and identical postprocessing.
2. Report class-wise IoU, boundary quality, and small-object results separately from OA and mIoU.
3. Measure per-tile latency, peak memory, input transfer cost and end-to-end throughput on the intended deployment hardware.
4. For sidewalks, evaluate Tile2Net on high-resolution local orthophotos and an APE-style sidewalk/crossing reference; fine-tune only after measuring the zero-shot domain gap.
5. Treat the SAM LoRA job as pending until CUDA training, validation, held-out scoring and artifact checks are complete.
