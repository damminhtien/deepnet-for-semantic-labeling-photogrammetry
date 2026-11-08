# DINOv2 joint-run validation review, 2026-11-01

**Status: SUPERSEDED** — Final outcome is recorded in isprs-dinov2-joint-final-review-2026-11-01.md.

See the [final review](isprs-dinov2-joint-final-review-2026-11-01.md) for saved
checkpoints and test scores.

The joint DINOv2 run completed 20 epochs and 1,280 optimizer updates. Epoch 15
won on validation. This snapshot uses live logs from 16:35 SGT, during held-out
scoring.

## Measured comparison

| Measurement | Historical SAM `candidate-b-r4-p384` | Current DINOv2 joint run |
| --- | ---: | ---: |
| Selected epoch | 30 | 15 |
| Potsdam validation mIoU | 74.3216% | 51.0339% |
| Potsdam validation overall accuracy | 89.8861% | 77.9859% |
| Potsdam validation scored pixels | 199,409,714 | 199,409,714 |
| Vaihingen validation mIoU | Not evaluated by this candidate | 47.6903% |
| Selection score | Potsdam validation mIoU | Mean of Potsdam/Vaihingen: 49.3621% |
| Potsdam held-out test mIoU | 73.7024% | Pending |

The Potsdam validation gap is **23.2877 percentage points** in mIoU and
**11.9002 points** in overall accuracy. Current values are rounded live-log
observations; full-precision metrics await the terminal artifact download.

The historical validation scene IDs match the current Potsdam manifest:
`3_12`, `4_10`, `4_11`, `4_12`, `5_11`, `6_12`. Each scene's scored-pixel count
matches. Summing the historical confusion matrix rows gives current quality-report
truth supports exactly, for classes 0 through 5:

```text
56,474,309  67,405,057  38,412,461  29,005,158  3,090,146  5,022,583
```

The two runs match in validation scene IDs and truth counts; label-map byte
equality is unverified. SAM scores non-overlapping 512-pixel tiles; DINOv2 uses
504-pixel tiles at stride 252 with weighted overlap. Both mIoU scores cover six
classes and ignore label 6.

## Training budget and learning curve

| Full-run budget | Historical SAM | Current DINOv2 |
| --- | ---: | ---: |
| Optimizer updates | 13,650 | 1,280 |
| Microbatch / accumulation | 1 / 1 | 1 / 4 |
| Potsdam patch presentations | 13,650 | 2,560 |
| Vaihingen patch presentations | 0 | 2,560 |
| Potsdam crop | 384 x 384 | 504 x 504 |
| Potsdam input pixel presentations | 2,012,774,400 | 650,280,960 |
| Total trainable parameters | 618,182 | 11,777,170 |

DINOv2 ran 10.66× fewer optimizer updates with different accumulation, 5.33×
fewer Potsdam patches and 3.10× fewer Potsdam pixel presentations, counting
repeated and augmented samples. It trains 19.05× as many parameters across LoRA,
adapter and decoder.

At similar Potsdam exposure, SAM epoch 10 reached **68.8812%** validation mIoU
after 670,924,800 pixel presentations. DINOv2 epoch 20 reached **50.9364%**
after 650,280,960. DINOv2 also trained on Vaihingen.

| DINOv2 epoch | Potsdam mIoU | Vaihingen mIoU | Selection mean |
| --- | ---: | ---: | ---: |
| 1 | 9.9485% | 7.4428% | 8.6956% |
| 10 | 42.2523% | 40.3163% | 41.2843% |
| 15, selected | 51.0339% | 47.6903% | 49.3621% |
| 20 | 50.9364% | 47.3334% | 49.1349% |

Validation improves around epochs 10–15 and plateaus through epoch 20. The
cosine schedule approaches zero learning rate; extending it is untested. The run
logs patch losses but does not score mIoU on the full training set.

Full-scene validation took 12,182.672 seconds (92.83%) of the 13,124.092 seconds
from the first update through epoch-20 validation. First-to-last update
intervals sum to 821.707 seconds, excluding each epoch's first update. The logs
do not time raster reads, overlap and decoding separately.

## Implementation findings and hypotheses

1. **The decoder receives unnormalized intermediate features.**
   `TransformerFeatureBackbone.forward_features` selects
   `hidden_states[3, 6, 9, 12]` as returned by Transformers 5.20.2.
   `Dinov2Model.forward` applies LayerNorm to `last_hidden_state`.
   Meta's reference
   [`get_intermediate_layers`](https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py#L278)
   applies the trained final LayerNorm to each selected output by default.
2. **The IRRG spectral projection starts randomly.** The three optical channels
   pass through a learned 1x1 convolution with Kaiming-uniform initialization,
   then ImageNet mean/std normalization. Test channel-preserving IRRG
   initialization to see whether the random mixture hurts a mostly frozen encoder.
3. **The decoder and adapter start from random weights.** DINOv2 refinement blocks
   use convolutions and GELU without normalization. SAM uses a pretrained neck
   and a smaller GroupNorm decoder. DINOv2 trains 19.05 times as many parameters
   with fewer Potsdam samples.
4. **The joint objective and sampling differ.** DINOv2 averages two Potsdam and
   two Vaihingen microbatch losses per update. SAM optimizes Potsdam alone.
   Negative transfer is possible but requires a Potsdam-only control. Historical
   sampling targets clutter for 50% of crops; current sampling uses a 30%
   class-aware mixture across all classes. Stronger affine/color augmentation,
   LoRA rank, weight decay and the selection objective also changed. Both paths
   use the same weighted CE/macro-Dice implementation and 0.7/0.3 coefficients.

Feature depths follow the Transformers embedding-plus-block convention. Cloud
inputs and LoRA preflight passed. Attempt 1 stopped before training on a torchao
compatibility error.

## Next decisions

Finish scoring the frozen checkpoint and audit hashes and per-scene confusion
counts. Compare class IoU and predicted versus actual pixel counts.

Test a Potsdam-only baseline with a simple dense head, reference feature
normalization and deterministic IRRG initialization. Verify gradients and
small-subset fit, then vary one component at a time at matched exposure and
scoring settings. Add Vaihingen after establishing the baseline. Budget
full-scene validation and optimizer updates before training.

The UniverSat, AnySat and TerraMind probes used frozen linear heads at 5 m
resolution. The direct Potsdam comparison here is with the tuned SAM recipe.

## Evidence and reproducible figure

Historical artifacts, verified directly for this review:

```text
/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/runs/
  sam-vitb-lora-kaggle-output-20261020/snapshot-20261020-retrain-04/
  sam-vitb-lora-potsdam-20261020-retrain-04/candidate-b-r4-p384/
  metrics.json, run_manifest.json, validation_log.jsonl
```

Current private kernel:
[`damminhtien/dinov2-joint-irrg-segmentation-20261101`](https://www.kaggle.com/code/damminhtien/dinov2-joint-irrg-segmentation-20261101).
Training config SHA-256 remains
`bfe34dd24bc2899b511271f11e288dc4f95e97f78dabbda54cf02fde7502f1ea`.
The uploaded runner wheel is frozen at source revision `885535f`; model source
is unchanged between that revision and this review.

External run root:
`/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/runs/isprs-dinov2-joint-v1-20261101-attempt-02`.
Under `analysis-20261101/`, `comparison.json` records the comparisons and caveats;
`dinov2-validation-from-live-log.json` contains all 20 observations;
`build_review.py` renders `validation-comparison.png` and `.pdf` using Matplotlib;
`evidence-sha256.json` binds the input snapshot and derived artifacts. The image
was rendered and visually inspected. These derived artifacts remain outside Git.
