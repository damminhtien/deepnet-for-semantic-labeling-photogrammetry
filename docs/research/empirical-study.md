# Understanding accuracy, errors, and compute in Potsdam aerial segmentation

**Status: CURRENT** — detailed supporting analysis; [`docs/results.md`](../results.md) is the canonical numerical summary.

## Abstract

This empirical study revisits four single-model Potsdam segmentation systems—SAM,
DINOv2, DINOv3, and PSPNet—to explain where their recorded scores change, how stable
the DINOv3–DINOv2 difference is across scenes, which tested recipe changes did not help,
and what quality costs in training and inference. It does not propose a new segmentation
architecture or claim a backbone-only causal effect. The 14-scene benchmark cohort has
appeared in previous experiments, so the results are historical system comparisons, not
a newly blind evaluation.

## Descriptive analysis questions

These questions break down the system-level comparison defined in
[`docs/research.md`](../research.md). They are descriptive subquestions; the separate
causal pipeline question remains open.

1. Which class confusions account for the recorded DINOv2-to-DINOv3 change?
2. Does the aggregate DINOv3 advantage remain when scenes are inspected separately or
   removed one at a time?
3. What do the tested bundled-feature and stronger-augmentation runs show, and what do
   they leave unidentified?
4. What quality and compute trade-offs were measured between matched RGB DINOv3 and
   PSPNet systems?
5. Where do two systems make different decisions at the same valid pixel, and which
   representative error patterns can be inspected directly?

## Dataset and evaluation

The Potsdam split contains 18 training scenes, six validation scenes, and 14 benchmark
scenes. Training uses participant labels; validation and benchmark scoring use the
eroded reference labels. The six scored classes are impervious surfaces, building, low
vegetation, tree, car, and clutter/background. Label ID 6 is ignored.

Every benchmark table in this report is reduced from archived per-scene confusion
matrices. The DINOv2/DINOv3 comparison covers 467,862,855 recorded valid pixels across
the same scene IDs. DINOv3 and PSPNet additionally share the RGB cache hash, split hash,
512×512 windows, stride 256, identity inference, and batch size 4. The source artifacts,
checksums, audits, and recomputation code are listed in
[`claims-and-evidence.md`](claims-and-evidence.md) and
[`data/potsdam-empirical-analysis.json`](data/potsdam-empirical-analysis.json).

## Systems in the comparison

| System | Input/recipe scope | Observed benchmark OA | Observed six-class mIoU |
| --- | --- | ---: | ---: |
| SAM ViT-B + LoRA | Historical multispectral recipe | 87.8142% | 73.7024% |
| DINOv2 ViT-B/14 | Historical IRRG complete recipe; 504 windows, stride 252 | 91.4785% | 79.6122% |
| PSPNet–ResNet101 | RGB, matched current cache and inference policy | 91.2834% | 79.4235% |
| DINOv3 ViT-B/16 | RGB, matched current cache and inference policy | 92.0397% | 81.0932% |

SAM and DINOv2 are historical recipe references. Their channels, training choices, and
tiling differ from the matched RGB systems. DINOv3 versus PSPNet is the closer system
comparison, but still does not isolate architecture from optimization or decoder
design. See [the claim ledger](claims-and-evidence.md).

## Findings

### Error distribution

DINOv3 has 2,625,662 fewer errors than DINOv2 in the recorded cohort, a 6.59% decrease
from the DINOv2 error count. Clutter/background contributes the largest class-level
change: +1,804,538 true positives, +9.0588 recall points, and +6.0635 IoU points, with
704,648 additional false positives. Cars gain recall but lose 0.2816 IoU points as
precision falls. The largest per-confusion reductions are clutter→impervious
(1,248,458) and tree→low-vegetation (926,205); low-vegetation→tree increases by 425,001.

These values compare aggregate confusion counts. They do not identify shared pixels or
causes. The full cell-by-cell and class-level breakdown is in
[`error-decomposition.md`](error-decomposition.md).

### Scene consistency

DINOv3 improves OA and mIoU on 12 of 14 scenes. Its smallest OA occurs on scene `4_14`
(85.9410%); the largest OA gain is on `2_14` (+1.6061 points), while the largest mIoU
gain is on `6_13` (+5.1022). Leave-one-scene-out mIoU differences remain positive from
1.2636 to 1.6526 points. The scene-cluster bootstrap and correlation summaries are
descriptive; 14 spatially related scenes do not support treating hundreds of millions
of pixels as independent observations. Details are in
[`scene-level-robustness.md`](scene-level-robustness.md).

### Negative results

The RGB-only DINOv3 selection arm A exceeds bundled-feature arm B by 0.1401 validation
mIoU points and stronger-augmentation arm C by 0.2382 points. C raises car IoU by
0.1043 points while lowering clutter IoU by 1.1913 points. The same A recipe differs
by 0.2902 mIoU points between the seed-42 selection and seed-43 confirmation. The
evidence favors A under the tested selection rule, but does not establish that
multimodal inputs or stronger augmentation are generally harmful. The bundled features
and augmentation factors were not ablated separately. See
[`augmentation-and-modality-negative-results.md`](augmentation-and-modality-negative-results.md).

### Quality and compute

On the matched RGB cohort, DINOv3 exceeds PSPNet by 0.7563 OA points and 1.6697 mIoU
points. Full-scene scoring takes 405.23 seconds for DINOv3 and 254.83 seconds for
PSPNet; forward time accounts for 278.33 and 110.01 seconds respectively. PSPNet's
selection-plus-refit cost is 1.7235 allocated GPU-hours. DINOv3's winning arm A plus
refit is 3.5291 GPU-hours, while the full A/B/C/D selection and refit is 12.7705
GPU-hours. The distinction is necessary because the DINOv3 campaign evaluated more
candidate recipes. Peak VRAM is reported only after the supplemental paired inference
reproduces both archived confusion matrices. The campaign accounting scope is
[CLAIM-009](../../artifacts/public/evidence-manifest.json). See
[`quality-cost-analysis.md`](quality-cost-analysis.md).

### Paired decisions and failure examples

The supplemental Kaggle job produced prediction maps with inference only. Both models
reproduced every archived per-scene confusion matrix exactly, and a local pass over the
hashed held-out cache independently reproduced the same matrices on all 467,862,855
scored pixels. Both models are correct on 417,385,769 pixels and wrong on 27,547,977.
DINOv3 is correct where PSPNet is wrong on 13,233,866 pixels; PSPNet is correct where
DINOv3 is wrong on 9,695,243. The net difference, 3,538,623 correctly classified
pixels, equals the aggregate accuracy difference. The systems agree on 94.51% of
scored pixels.

The paired class counts favor DINOv3 most on low vegetation (+1,387,660 net correct
pixels) and clutter/background (+1,302,537); PSPNet has small paired correctness
advantages for building and car pixels despite DINOv3's higher aggregate IoU for every
class. The [failure taxonomy and case
studies](failure-taxonomy-and-case-studies.md) describe the shared errors, class
transitions, and 20 deterministically selected crops. The reference remains eroded, so
the examples do not measure full-boundary accuracy. The paired run also supplies the
CUDA memory measurements reported in [quality and
cost](quality-cost-analysis.md).

## Limitations and next experiments

- The 14 benchmark scenes were observed in earlier campaigns; this is not a fresh blind
  test or unseen-city result.
- DINOv2 and DINOv3 differ in more than backbone version. Their score difference is not
  a causal backbone effect.
- A/B/C use one selection seed; only A has one additional confirmation seed. No
  multi-seed treatment estimate is available.
- The auxiliary branch bundles IR, spectral index, relief, and DSM availability; the
  individual input effects remain unknown.
- Eroded references exclude some boundary pixels, so they cannot establish full-edge
  boundary quality.
- The quality–cost frontier contains only the two matched RGB systems and the recorded
  Kaggle hardware and policy.

The study's contribution is a reproducible characterization of recorded single-model
systems: where class errors move, how the comparison behaves across scenes, what tested
changes failed to improve the selected metric, and how the measured accuracy gain trades
against compute. It is evidence to guide controlled future work, not a claim of a new
algorithm.
