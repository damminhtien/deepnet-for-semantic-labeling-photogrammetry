# Segmentation model and sidewalk research notes

Date checked: 2026-10-19.

## Potsdam leaderboard context

The ISPRS Potsdam 2D table lists the Southwest Jiaotong University entry as
`SWJ`, with an Overall score of 89.5%. The per-class columns report F1 on eroded
references.

`AMA_1` scored 91.2% Overall and `HUSTW3`/`HUSTW4` scored 91.6%. The AMA method
note describes a five-fold RefineNet ensemble. The HUST note describes its
Vaihingen entry: a supervised deconvolution/U-Net approach with IRRG, DSM, nDSM
and label voting.

Sources: [official Potsdam results table](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/results/label_challenge_2D_potsdam_result_ifr.html), [SWJ per-tile details](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/results/2D_labeling_potsdam/2D_labeling_Potsdam_details_SWJ/index.html), [AMA_1 method description](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/results/papers/Method_Description_AMA.pdf), and [HUST brief method description](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/results/papers/HUST_WH_brief_descr.pdf).

## Provisional flagship candidate

Prototype TerraMind and keep UniverSat for comparison. TerraMind leads sampled
RGB mIoU on LandCover.ai and LoveDA and held-out OA on Potsdam. UniverSat leads
held-out mIoU on Potsdam and Vaihingen. Cross-domain mIoU is low for all three
encoders.

The [RGB probes](../experiments/frozen-encoder-probes-2026-10-19.md) and
[Potsdam/Vaihingen
evaluation](../experiments/potsdam-vaihingen-foundation-probe-2026-10-19.md) use
frozen encoders on coarse grids. Their timings include feature extraction,
repeated fits and evaluation. Full-resolution tile latency, peak memory and
fine-tuned accuracy remain unmeasured in these probes.

## SAM status

The project adapts SAM ViT-B with rank-4 LoRA and a supervised six-class
decoder. As of 2026-10-19, the code and private Kaggle job were prepared;
training and held-out scores were pending.

Sources: [official Segment Anything repository](https://github.com/facebookresearch/segment-anything) and [SAM paper](https://arxiv.org/abs/2304.02643).

## Pedestrian paths and sidewalks

Tile2Net maps roads, sidewalks, crosswalks and footpaths from aerial imagery
into polygons and a pedestrian network. Its 2023 paper reports 82.67% sidewalk
IoU, 75.42% crosswalk IoU and 84.51% four-class mIoU for the model evaluated
there. The beta release described in October 2026 uses updated weights.

The October 2026 Tile2Net guide specifies CUDA and Python 3.11–3.14. It
recommends orthorectified imagery at zoom 19 or higher and sets zoom 18 as the
minimum. Measure the effects of canopy, shadows and resolution on local imagery
before fine-tuning.

The [APE dataset](https://sidewalks.washington.edu/2024/03/01/ape/) pairs aerial
imagery and street maps with human-validated sidewalk, crossing and corner-bulb
masks. Potsdam and Vaihingen group sidewalks and roads under impervious
surfaces.

Sources: [Tile2Net repository and current runtime requirements](https://github.com/VIDA-NYU/tile2net), [Tile2Net project preparation guide](https://github.com/VIDA-NYU/tile2net/blob/main/BASICS.md), [2023 Tile2Net paper](https://fmiranda.me/publications/pdfs/tile2net.pdf), and [University of Washington APE dataset description](https://sidewalks.washington.edu/2024/03/01/ape/).

## Next evaluation that would resolve the remaining choice

1. Run TerraMind and UniverSat on the same full-resolution, held-out aerial scenes with tiled inference and identical postprocessing.
2. Report class-wise IoU, boundary quality, and small-object results separately from OA and mIoU.
3. Measure per-tile latency, peak memory, input transfer cost and end-to-end throughput on the intended deployment hardware.
4. For sidewalks, evaluate Tile2Net on high-resolution local orthophotos and an APE-style sidewalk/crossing reference; fine-tune only after measuring the zero-shot domain gap.
5. Treat the SAM LoRA job as pending until CUDA training, validation, held-out scoring and artifact checks are complete.
