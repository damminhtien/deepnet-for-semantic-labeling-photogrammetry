# DINOv2 joint-run validation review, 2026-11-01

The first successful DINOv2 joint training attempt has completed 20 epochs and
1,280 optimizer updates. Its selected epoch 15 is substantially weaker on Potsdam
validation than the audited historical SAM candidate. This establishes a gap
between the two complete training recipes, not an isolated backbone comparison.
Held-out scoring is still running at this review's 16:35 SGT observation; no
DINOv2 test metric or per-class diagnosis is claimed here.

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

These checks support comparison of the observed Potsdam task scores. They do not
prove bitwise equality of the historical and current label maps. The evaluation
recipes also differ: historical SAM uses non-overlapping 512-pixel tiles; DINOv2
uses 504-pixel tiles, stride 252, with weighted overlap. A controlled architecture
comparison would require the same evaluation implementation and frozen protocol.
The six-class, ignore-ID-6 mIoU gap is not explained by the earlier macro-F1 fix.

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

DINOv2 has 10.66 times fewer global optimizer updates, but accumulation differs.
The more useful exposure measures are 5.33 times fewer Potsdam patches and
3.10 times fewer Potsdam input pixel presentations. Presentations include repeated
and augmented pixels; they are not distinct data coverage or equal information.
The trainable-parameter ratio is 19.05 across the entire model, including LoRA,
adapter and decoder; it is not the decoder-only ratio.

Budget is an incomplete explanation. At approximately matched Potsdam exposure,
SAM epoch 10 reached **68.8812%** validation mIoU after 670,924,800 input pixel
presentations. DINOv2 epoch 20 reached **50.9364%** after 650,280,960. Other recipe
differences remain confounded, including the additional Vaihingen exposure.

| DINOv2 epoch | Potsdam mIoU | Vaihingen mIoU | Selection mean |
| --- | ---: | ---: | ---: |
| 1 | 9.9485% | 7.4428% | 8.6956% |
| 10 | 42.2523% | 40.3163% | 41.2843% |
| 15, selected | 51.0339% | 47.6903% | 49.3621% |
| 20 | 50.9364% | 47.3334% | 49.1349% |

Early validation is unstable, followed by improvement around epochs 10--15 and a
plateau through epoch 20. A cosine schedule approaching zero learning rate makes
simply extending the same schedule an untested remedy. No full-training-set mIoU
was measured, so these curves alone do not establish overfitting. Sparse losses
from different training patches cannot establish a train/validation accuracy gap.

The log contains 12,182.672 seconds of full-scene validation, 92.83% of the
13,124.092 seconds from the first optimizer update through epoch-20 validation.
The sum of first-to-last update intervals per epoch is 821.707 seconds; it excludes
the first update of each epoch. Thus the run's elapsed hours mostly measure
evaluation cost, not optimization effort. Overlapping inference, dense decoding
and raster/memmap work are all involved; their individual costs were not profiled.

## Implementation findings and hypotheses

1. **Intermediate-feature normalization differs from the reference path.**
   `TransformerFeatureBackbone.forward_features` directly selects
   `hidden_states[3, 6, 9, 12]`. In installed Transformers 5.20.2,
   `Dinov2Model.forward` normalizes `last_hidden_state`, while returning the
   encoder's original `hidden_states`. The repo does not normalize those selected
   feature tensors before decoding. Meta's reference
   [`get_intermediate_layers`](https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py#L278)
   defaults to `norm=True` and applies its trained final LayerNorm to every
   selected output. This is a concrete feature-interface difference and a strong
   ablation candidate; its accuracy effect has not been measured.
2. **The IRRG spectral projection starts randomly.** The three optical channels
   pass through a learned 1x1 convolution with Kaiming-uniform initialization,
   then ImageNet mean/std normalization. Input normalization exists. A random
   spectral mixture can initially disrupt the pretrained encoder's expected
   statistics and channel structure, especially with mostly frozen weights.
   A deterministic IRRG channel-preserving initialization is a proposed control,
   not an established physically correct conversion from IRRG to RGB.
3. **Much more of the dense prediction path starts from scratch.** DINOv2's
   custom multi-scale decoder and input adapter are randomly initialized. Its
   refinement blocks use convolutions and GELU without normalization; the older
   SAM path uses its pretrained neck and a smaller GroupNorm decoder. The whole
   DINOv2 model trains 19.05 times as many parameters with less Potsdam exposure.
   This raises an optimization concern; it does not prove a capacity problem or
   prove that adding normalization will recover the observed gap.
4. **The joint objective and sampling differ.** DINOv2 averages two Potsdam and
   two Vaihingen microbatch losses per update. SAM optimizes Potsdam alone.
   Negative transfer is possible but requires a Potsdam-only control. Historical
   sampling targets clutter for 50% of crops; current sampling uses a 30%
   class-aware mixture across all classes. Stronger affine/color augmentation,
   LoRA rank, weight decay and the selection objective also changed. Both paths
   use the same weighted CE/macro-Dice implementation and 0.7/0.3 coefficients.

The configured feature depths match the Hugging Face embedding-plus-block hidden
state convention; this review found no evidence of an off-by-one block selection.
Cloud input hashes and the actual LoRA preflight passed. Attempt 1's torchao
compatibility failure happened before training and produced no model result;
it cannot itself explain attempt 2's lower accuracy.

## Next decisions

Finish the already-frozen held-out evaluation and audit its selected checkpoint,
hashes and per-scene confusion counts. Use class IoU and prediction/truth support
to identify which classes lose accuracy; no current class-collapse claim is yet
supported. Do not change this run or choose a replacement using its test score.

For a new experiment, predeclare a Potsdam-only baseline with a simple dense head,
the reference normalized features and deterministic IRRG initialization. Verify
gradient flow and ability to fit a small training subset, then compare individual
changes on validation with matched exposure, evaluation and schedule. Add
Vaihingen only after that baseline is stable. A future budget can spend fewer
full-scene validation passes and more optimizer updates, with its selection
schedule declared before training. These are proposed experiments, not launched
runs or proven fixes.

The older UniverSat/AnySat/TerraMind probes used coarse 5 m grids and frozen linear
heads; their scores are not native-resolution architecture controls. The current
DINOv2 score is not worse than every historical experiment. The strongest
supported comparison is the substantial gap to the tuned SAM Potsdam recipe.

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
