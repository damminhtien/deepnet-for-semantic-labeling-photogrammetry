# Research questions and contribution

## Scope

This repository studies geospatial and aerial semantic segmentation through one shared
pipeline and a small set of audited Potsdam experiments. The primary question answered
by the current study is:

> What accuracy and compute trade-offs do recorded single-model segmentation systems
> exhibit under a documented, scene-aware evaluation protocol on ISPRS Potsdam?

The study compares complete trained systems. It does not propose a new segmentation
algorithm and does not estimate the isolated causal effect of an encoder, modality,
augmentation, or decoder.

A second question remains open:

> Which data preparation, augmentation, and modality choices causally affect segmentation
> performance when model, split, sampling, training budget, and evaluation are controlled?

The current A/B/C validation runs provide narrow observations about tested recipes, not
a complete answer to this question. A proposed controlled study is recorded as
[`CONFIGURED_NOT_RUN`](experiments/future-study-design.md).

## Contributions

1. **Geospatial data integrity.** The data preparation work records scene roles, source
   checksums, band identity, raster grids, label support, and auxiliary-data alignment.
   The Potsdam RGB manifest and data profile document how the verified RGB bands differ
   from the legacy IRRG channel naming. This contribution is an auditable engineering
   contract, not a new remote-sensing method. See the [data profile](research/potsdam-vaihingen-data-profile-2026-11-07.md)
   and [pipeline guide](data-pipeline.md).
2. **Experiment infrastructure.** The repository keeps scene splits, input views,
   preprocessing, sampling, evaluation, and run protocols in versioned configurations.
   The public matrix bundle and evidence registry allow an outside reader to recompute
   the headline Potsdam metrics without access to author-local runs. This currently
   establishes metric reproduction (R1); see [reproducibility](reproducibility.md).
3. **Empirical system comparison.** The experiments document complete DINOv2, DINOv3,
   PSPNet, and SAM recipes, then analyze validation choices, per-class errors, scene
   variation, paired predictions, and recorded runtime. The [results page](results.md)
   is the sole numerical summary; supporting analysis is indexed in
   [the empirical study](research/empirical-study.md).

## Baseline positioning

The comparison covers established system families: SAM's promptable segmentation model
([paper](https://arxiv.org/abs/2304.02643)), the self-supervised DINOv2 and DINOv3
representations ([DINOv2 paper](https://arxiv.org/abs/2304.07193),
[DINOv3 paper](https://arxiv.org/abs/2508.10104)), and PSPNet's pyramid scene context
aggregation ([CVPR paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.html)).
The Potsdam experiments adapt these systems with task-specific heads or LoRA; they are
not reproductions of the papers' original benchmarks. Their role is to provide
system-level reference points for the accuracy and runtime study, not a controlled
architecture benchmark. Model specifications and data paths are listed in the
[architecture guide](architecture.md) and [experiment index](experiments/README.md).

## What the evidence shows

- The RGB-only DINOv3 selection recipe scored higher on the recorded validation objective
  than the tested bundled auxiliary-feature and stronger-augmentation recipes. The
  comparison does not isolate individual bands or augmentation operations
  ([CLAIM-007](../artifacts/public/evidence-manifest.json)).
- DINOv3 RGB scored higher than PSPNet RGB on the shared historical benchmark setup and
  required more recorded full-scene scoring time. This is a comparison of complete
  recipes on the measured hardware ([CLAIM-005](../artifacts/public/evidence-manifest.json)).
- Paired DINOv3 and PSPNet predictions show both corrections and regressions. Many
  errors are shared, and the largest class-level differences involve low vegetation
  and clutter/background. The maps describe model decisions; they do not establish
  their physical or semantic causes ([CLAIM-006](../artifacts/public/evidence-manifest.json)).
- DINOv3 has higher OA than DINOv2 on 12 of the 14 recorded scenes. Scene-cluster and
  leave-one-out analyses remain descriptive for this small, spatially related cohort
  ([CLAIM-010](../artifacts/public/evidence-manifest.json)).
- GPU-hour comparisons depend on the scope of the candidate search. The full DINOv3
  selection campaign and the PSPNet selection phase are different accounting units
  ([CLAIM-009](../artifacts/public/evidence-manifest.json)).

The four benchmark rows are single-model results. No ensemble is used. The scope of
these findings and unresolved validity risks are collected in
[`limitations.md`](limitations.md).

## Reading order

Start with [results](results.md) for recomputed metrics, then read the
[experiment index](experiments/README.md) for protocol status. The
[empirical study](research/empirical-study.md) links the detailed class-confusion,
scene-robustness, modality, paired-failure, and quality-cost analyses. Implementation
interfaces are documented in [architecture](architecture.md), and raster preparation
is documented in [data pipeline](data-pipeline.md).
