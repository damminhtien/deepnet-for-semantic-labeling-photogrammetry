# Experiment and evidence index

Detailed campaign notes preserve their original phase records. The status line in each
report describes how to use it today; it does not rewrite the historical result.

## Canonical research documents

| Purpose | Document |
| --- | --- |
| Research questions and contributions | [Research](../research.md) |
| Public benchmark results | [Results](../results.md) |
| Validity and scope | [Limitations](../limitations.md) |
| Public verification steps | [Reproducibility](../reproducibility.md) |
| System interfaces | [Architecture](../architecture.md) |
| Data preparation | [Data pipeline](../data-pipeline.md) |
| Figure sources | [Figures](../figures.md) |

## Completed campaigns

| Status | Report | Use |
| --- | --- | --- |
| CURRENT | [DINOv3 RGBIR v2 final artifact review](potsdam-dinov3-rgbir-v2-final-review-2026-11-06.md) | Canonical DINOv3 campaign outcome and benchmark provenance. |
| CURRENT | [DINOv3 selection summary](potsdam-dinov3-rgbir-v2-selection-summary-2026-11-06.md) | Selected validation recipe and seed-43 confirmation. |
| CURRENT | [DINOv3 campaign protocol](potsdam-dinov3-rgbir-v2.md) | Frozen campaign protocol and configuration references. |
| CURRENT | [PSPNet RGB comparison](potsdam-pspnet-rgb-v1.md) | Canonical PSPNet comparison run. |
| HISTORICAL | [DINOv2 coverage refit](potsdam-dinov2-coverage-refit.md) | Completed earlier DINOv2 selection/refit protocol. |
| HISTORICAL | [DINOv2 final artifact review](potsdam-dinov2-coverage-refit-final-review-2026-11-05.md) | Earlier DINOv2 checkpoint and evaluation evidence. |
| HISTORICAL | [DINOv2 joint final review](isprs-dinov2-joint-final-review-2026-11-01.md) | Separate Potsdam/Vaihingen protocol. |
| HISTORICAL | [Frozen encoder probes](frozen-encoder-probes-2026-10-19.md) | Earlier representation probe; not a direct comparison to final recipes. |
| HISTORICAL | [Foundation encoder evaluation](potsdam-vaihingen-foundation-probe-2026-10-19.md) | Earlier cross-dataset probes and metric artifact. |

## Superseded phase records

| Status | Report | Superseded by |
| --- | --- | --- |
| SUPERSEDED | [DINOv2 joint validation review](isprs-dinov2-joint-validation-review-2026-11-01.md) | [DINOv2 joint final review](isprs-dinov2-joint-final-review-2026-11-01.md) |
| SUPERSEDED | [DINOv3 verified GPU preflight](potsdam-dinov3-rgbir-v2-preflight-2026-11-05.md) | [DINOv3 final review](potsdam-dinov3-rgbir-v2-final-review-2026-11-06.md) |
| SUPERSEDED | [DINOv3 selection review, 2026-11-05](potsdam-dinov3-rgbir-v2-selection-review-2026-11-05.md) | [DINOv3 selection summary](potsdam-dinov3-rgbir-v2-selection-summary-2026-11-06.md) |
| SUPERSEDED | [DINOv3 selection review, 2026-11-06](potsdam-dinov3-rgbir-v2-selection-review-2026-11-06.md) | [DINOv3 selection summary](potsdam-dinov3-rgbir-v2-selection-summary-2026-11-06.md) |

## Not run

| Status | Report | Note |
| --- | --- | --- |
| CONFIGURED_NOT_RUN | [Controlled pipeline study design](future-study-design.md) | Proposed ablations and independent evaluation; no jobs have been launched. |
| CONFIGURED_NOT_RUN | [Segmentation model and sidewalk research note](../research/segmentation-model-and-sidewalk-research-2026-10-19.md) | Research direction only; not evidence for Potsdam model rankings. |

## Status definitions

- **CURRENT** identifies the canonical record for an executed comparison or completed
  phase.
- **HISTORICAL** identifies a completed result retained for context and traceability.
- **SUPERSEDED** identifies an interim record whose pending or selection conclusion was
  finalized by a later report; the original contents remain unchanged apart from this
  status annotation.
- **CONFIGURED_NOT_RUN** identifies a protocol or proposal without completed execution.
