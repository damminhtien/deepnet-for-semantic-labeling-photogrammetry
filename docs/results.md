# Potsdam benchmark results

This page is the canonical numerical summary. Its generated tables are recomputed from
public, per-scene confusion matrices and the checksums in the evidence registry. The
14-scene cohort has been examined in prior experiments; see the [study scope and
limitations](limitations.md#benchmark-reuse-and-external-validity).

See the [benchmark metric formulas](benchmark-metrics.md) for the exact definitions.

<!-- BEGIN GENERATED METRICS -->

### Aggregate results

Metrics are recomputed by summing the 14 per-scene confusion matrices for each model. Rows are complete recorded systems; only DINOv3 RGB and PSPNet RGB share the matched input, split, eroded reference, $512 \times 512$ windows, stride 256, batch size, and identity inference recipe. The benchmark was examined in earlier campaigns.

| Model | OA | Six-class mIoU | Errors / scored pixels | Recorded full-scene scoring | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| SAM ViT-B + LoRA | 87.8142% | 73.7024% | 57,012,689 / 467,862,855 | Not reported in public metric artifact | CLAIM-004 |
| DINOv2 ViT-B/14 refit | 91.4785% | 79.6122% | 39,868,882 / 467,862,855 | Not reported in public metric artifact | CLAIM-003 |
| PSPNet–ResNet101 RGB refit | 91.2834% | 79.4235% | 40,781,843 / 467,862,855 | 254.83 s | CLAIM-002 |
| DINOv3 ViT-B/16 RGB refit | 92.0397% | 81.0932% | 37,243,220 / 467,862,855 | 405.23 s | CLAIM-001 |

OA is overall accuracy. mIoU is the unweighted mean of the six class IoUs. Scoring excludes label ID 6. Runtime is included only where it was recorded in the source artifact; it is not imputed.

### Per-class IoU and F1

Each cell reports IoU / F1 in percent, recomputed from the same aggregate matrix.

| Class | SAM IoU / F1 | DINOv2 IoU / F1 | PSPNet IoU / F1 | DINOv3 IoU / F1 |
| --- | ---: | ---: | ---: | ---: |
| Impervious surfaces | 83.2068% / 90.8338% | 87.8091% / 93.5089% | 87.9163% / 93.5697% | 88.7372% / 94.0325% |
| Building | 86.2614% / 92.6240% | 94.6559% / 97.2546% | 94.1508% / 96.9873% | 95.0554% / 97.4650% |
| Low vegetation | 74.2724% / 85.2371% | 79.0312% / 88.2876% | 78.1639% / 87.7438% | 80.0577% / 88.9245% |
| Tree | 76.3332% / 86.5783% | 81.0786% / 89.5507% | 80.7474% / 89.3484% | 81.8287% / 90.0064% |
| Car | 85.1687% / 91.9904% | 92.7234% / 96.2243% | 92.2606% / 95.9745% | 92.4418% / 96.0725% |
| Clutter/background | 36.9718% / 53.9846% | 42.3747% / 59.5256% | 43.3019% / 60.4345% | 48.4383% / 65.2638% |

### Per-scene OA and mIoU

Each cell reports OA / mIoU in percent from that scene's confusion matrix.

| Scene | SAM OA / mIoU | DINOv2 OA / mIoU | PSPNet OA / mIoU | DINOv3 OA / mIoU |
| --- | ---: | ---: | ---: | ---: |
| 2_13 | 87.7361% / 70.6263% | 92.2660% / 79.3416% | 91.3616% / 77.7808% | 92.7240% / 80.0330% |
| 2_14 | 89.9088% / 71.8496% | 91.7304% / 78.1964% | 92.3210% / 78.1169% | 93.3365% / 79.4943% |
| 3_13 | 90.1316% / 73.5960% | 92.4143% / 79.3062% | 92.0192% / 78.7051% | 92.6437% / 79.6974% |
| 3_14 | 88.1053% / 72.9091% | 91.5149% / 79.4884% | 90.5486% / 78.1545% | 91.7142% / 80.3907% |
| 4_13 | 88.4252% / 71.2480% | 90.8186% / 75.2625% | 90.6583% / 75.0235% | 90.9456% / 75.6824% |
| 4_14 | 83.3686% / 71.3801% | 84.5825% / 72.7476% | 85.0377% / 74.0926% | 85.9410% / 75.6118% |
| 4_15 | 85.6430% / 67.7694% | 92.7802% / 77.4077% | 92.1651% / 76.8014% | 93.4261% / 78.7794% |
| 5_13 | 89.4877% / 72.6255% | 92.3054% / 78.8640% | 91.9218% / 77.1074% | 92.1974% / 78.4836% |
| 5_14 | 86.7402% / 72.6363% | 89.7836% / 78.2548% | 89.8406% / 78.1301% | 89.7140% / 78.3203% |
| 5_15 | 87.4550% / 74.7227% | 90.3011% / 78.2573% | 90.8708% / 78.9757% | 91.5754% / 80.9597% |
| 6_13 | 89.3959% / 70.8955% | 93.8505% / 79.0186% | 94.2666% / 82.0338% | 94.7947% / 84.1208% |
| 6_14 | 90.4422% / 73.2448% | 94.1487% / 80.1941% | 93.8286% / 78.1579% | 94.7824% / 83.7827% |
| 6_15 | 86.9206% / 74.5411% | 93.3275% / 84.7020% | 92.1693% / 82.4166% | 93.3710% / 84.6902% |
| 7_13 | 85.6214% / 70.8983% | 90.8202% / 77.2034% | 90.8642% / 77.2959% | 91.3106% / 78.4851% |

### Matched RGB scene comparison

DINOv3 minus PSPNet is positive on 12 of 14 scenes. These paired recipe-level results do not isolate an architecture-only effect. Full-scene latency was 405.23 s for DINOv3 and 254.83 s for PSPNet in the archived benchmark profiles (CLAIM-005).

| Scene | DINOv3 OA | PSPNet OA | DINOv3 − PSPNet |
| --- | ---: | ---: | ---: |
| 2_13 | 92.7240% | 91.3616% | +1.3624 pp |
| 2_14 | 93.3365% | 92.3210% | +1.0155 pp |
| 3_13 | 92.6437% | 92.0192% | +0.6245 pp |
| 3_14 | 91.7142% | 90.5486% | +1.1656 pp |
| 4_13 | 90.9456% | 90.6583% | +0.2873 pp |
| 4_14 | 85.9410% | 85.0377% | +0.9033 pp |
| 4_15 | 93.4261% | 92.1651% | +1.2611 pp |
| 5_13 | 92.1974% | 91.9218% | +0.2756 pp |
| 5_14 | 89.7140% | 89.8406% | -0.1265 pp |
| 5_15 | 91.5754% | 90.8708% | +0.7045 pp |
| 6_13 | 94.7947% | 94.2666% | +0.5281 pp |
| 6_14 | 94.7824% | 93.8286% | +0.9538 pp |
| 6_15 | 93.3710% | 92.1693% | +1.2017 pp |
| 7_13 | 91.3106% | 90.8642% | +0.4463 pp |

### Recomputed comparisons

- DINOv3 vs PSPNet: OA +0.7563 pp, six-class mIoU +1.6697 pp (matched RGB recipes; CLAIM-005).
- DINOv3 vs historical DINOv2: OA +0.5612 pp, six-class mIoU +1.4810 pp; this compares complete recipes with different inputs and tiling (CLAIM-003).

The evidence bundle stores 467,862,855 scored pixels per model. Per-scene and aggregate matrices are in `artifacts/public/metrics/potsdam-benchmark-confusions.json`.

<!-- END GENERATED METRICS -->

For interpretation of paired predictions, scene variation, and failure examples, see
the [empirical study](research.md) and its linked analyses. The [experiment index](experiments/README.md)
identifies each detailed report's status.
