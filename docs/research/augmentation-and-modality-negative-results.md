# DINOv3 modality and augmentation results

## Full-validation comparison

The table reports the six-scene full validation at update 6,912 for each selection arm.
These checkpoints were selected under the original protocol; the table is descriptive
of the tested configurations and does not estimate a general treatment effect.

| Arm | Recipe | Seed | OA | mIoU | Errors |
| --- | --- | ---: | ---: | ---: | ---: |
| A | RGB, baseline augmentation | 42 | 93.0441% | 80.7498% | 13,870,767 |
| B | A plus bundled IR, spectral index, DSM relief, DSM mask | 42 | 92.9923% | 80.6097% | 13,973,954 |
| C | A with stronger geometric and optical augmentation | 42 | 92.9690% | 80.5115% | 14,020,417 |
| D | A recipe, confirmation seed | 43 | 92.9477% | 80.4596% | 14,062,959 |

B is 0.1401 mIoU points below A and has 103,187 more validation errors. The A-seed42
versus D-seed43 mIoU gap is 0.2902 points, more than twice the A–B difference. The
single confirmation seed shows that seed variation is material at this scale, but two
seeds are insufficient to estimate a seed distribution.

C is 0.2382 mIoU points below A and has 149,650 more errors. Its car IoU increases by
0.1043 points while clutter IoU falls by 1.1913 points. This is a measured
class-specific trade-off. A possible texture/context sensitivity in clutter is a
hypothesis only; the experiment does not isolate that mechanism.

## Test-time augmentation

Horizontal-flip TTA improves OA by 0.07858 points on seed 42 and 0.068995 points on seed
43. The locked threshold was 0.1 point, so neither check qualified. Identity inference
was retained. The archived TTA rows and their checksums are recorded in the analysis
provenance and the [DINOv3 final review](../experiments/potsdam-dinov3-rgbir-v2-final-review-2026-11-06.md).

## Limits on the interpretation

- B groups IR, spectral index, DSM relief, and the DSM mask. There is no evidence here
  that ranks those inputs individually.
- A/B/C each use one seed. The A confirmation uses one additional seed; it is not a
  replicated factorial study.
- Augmentation strength changed geometry and optical perturbations together. The
  experiment cannot identify which transform or probability explains the class changes.
- No conclusion follows that multimodal inputs or stronger augmentation are generally
  ineffective. The evidence is limited to these recipes, training budgets, and six
  validation scenes.

The numeric source tables are
[`selection arms`](data/potsdam-negative-results-arms.csv) and
[`contrasts`](data/potsdam-negative-results-contrasts.csv). Recreate them with:

```bash
uv run --locked --extra notebooks python scripts/research/potsdam_empirical.py \
  --runs-root /path/to/runs
```
