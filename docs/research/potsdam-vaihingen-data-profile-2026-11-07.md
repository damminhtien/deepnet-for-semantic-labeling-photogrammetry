# Potsdam and Vaihingen: Data Profile for ISPRS Urban Semantic Segmentation

**Scientific data report**<br>
**Prepared:** 7 November 2026<br>
**Scope:** Local scene catalogs, prepared split manifests, label composition, spatial support, and archived transfer evidence

## Abstract

Potsdam and Vaihingen share the ISPRS six-class urban labeling ontology, but their spatial sampling, scene geometry, class prevalence, and local modality coverage differ. This report profiles the prepared data currently present in the repository and relates those differences to archived model evaluations. It finds that one Vaihingen pixel covers 3.24 times the ground area of one Potsdam pixel; clutter/background represents 5.61% of Potsdam training labels and 0.88% of Vaihingen training labels; and Vaihingen clutter occurs in only 4 of 12 training scenes. Historical frozen-encoder probes also show a large cross-domain performance drop on an approximately 5 m grid. These observations support scene-aware sampling and ground-scale-aware evaluation. They do not establish that any one dataset is intrinsically harder, because the experiments differ in protocol and the urban scenes differ in more than class balance.

## 1. Introduction and scope

The ISPRS benchmarks provide very high resolution airborne orthophotos and six semantic classes. The official descriptions characterize Vaihingen as a smaller village with detached buildings and Potsdam as a denser historic city with larger blocks and narrow streets. The class ontology is shared, but the spatial context and sensing scale are not [ISPRS overview](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx).

This report answers four descriptive questions:

1. How do local scene geometry, ground sampling distance, and prepared data coverage differ?
2. How do six-class label proportions differ across train, validation, and test roles?
3. Which minority classes are represented across scenes, rather than only in aggregate pixel counts?
4. What does the archived cross-domain evidence say about transfer between the two datasets?

No model training or inference was run for this report. The current prepared manifests are the source for data counts. Archived experiments are included as qualified context and are not reinterpreted as fresh, untouched test results.

## 2. Data sources and acquisition geometry

| Property | Potsdam | Vaihingen |
|---|---|---|
| Official scene count | 38 | 33 |
| Ground sampling distance | 5 cm | 9 cm |
| Official tile geometry | 6,000 x 6,000 px; equal tiles of about 300 m x 300 m | Variable tile sizes; local area 3 is 3,007 x 2,006 px |
| Prepared optical view in this repository | 3-band IRRG | 3-band IRRG |
| Additional local modalities | RGBIR and DSM inventories each record 38 scenes | Official source describes a DSM, but the current local prepared view contains IRRG imagery and labels only |
| Local geospatial metadata | Prepared Potsdam scenes record EPSG:32633 | The local Vaihingen catalog retains pixel-local pairing; CRS and affine fields are absent |
| Current split roles | 18 train / 6 validation / 14 test | 12 train / 4 validation / 17 test |

The official Potsdam page specifies 38 equal 6,000 x 6,000 pixel patches at 5 cm GSD, with RGB, IRRG, RGBIR and DSM products [ISPRS Potsdam](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx). The Vaihingen page specifies 33 different-sized patches at 9 cm GSD and three 8-bit image bands corresponding to near infrared, red and green [ISPRS Vaihingen](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx).

The prepared optical input is IR, red, green in both local catalogs. The squared GSD ratio is `(9 / 5)^2 = 3.24`: a Vaihingen pixel covers 0.0081 m2, compared with 0.0025 m2 for a Potsdam pixel. Consequently, equal-sized pixel windows do not represent equal ground footprints. Patch sizes intended for cross-dataset comparison should be specified in metres and converted to pixels per dataset.

The figure below shows representative 45 m square training windows. The IRRG image is displayed as false color with infrared, red, and green mapped to display RGB. Each crop has its own 2nd-to-98th percentile display stretch; these colors are illustrative and are not calibrated reflectance values. The paired labels use the official six-class palette.

![Equal-ground-size IRRG and label examples for Potsdam and Vaihingen](figures/potsdam-vaihingen-scene-samples-scienceplots.png)

**Figure 1.** Representative training imagery and labels: Potsdam scene 2_10 and Vaihingen area 3. Each crop covers approximately 45 m x 45 m. Crops were selected to include diverse labels.

## 3. Data preparation and split coverage

Catalog validation succeeds for all 38 Potsdam scenes and all 33 Vaihingen scenes. Train, validation, and test scene IDs are disjoint in the prepared split manifests. The split unit is a scene; the current split definition is not region-held-out.

Area estimates below are computed from image pixel dimensions and the official GSD. “Image area” counts all image pixels in a role. “Scored area” multiplies the count of class IDs 0-5 by the squared GSD; eroded reference boundaries with ignore ID 6 are excluded.

| Dataset | Role | Scenes | Scored pixels | Image area (ha) | Scored reference area (ha) | Ignored image share |
|---|---:|---:|---:|---:|---:|---:|
| Potsdam | Train | 18 | 648,000,000 | 162.00 | 162.00 | 0.00% |
| Potsdam | Validation | 6 | 199,409,714 | 54.00 | 49.85 | 7.68% |
| Potsdam | Test | 14 | 467,862,855 | 126.00 | 116.97 | 7.17% |
| Vaihingen | Train | 12 | 58,554,459 | 47.43 | 47.43 | 0.00% |
| Vaihingen | Validation | 4 | 17,591,104 | 15.82 | 14.25 | 9.94% |
| Vaihingen | Test | 17 | 81,991,943 | 73.06 | 66.41 | 9.10% |

![Scene counts and ground coverage by split](figures/potsdam-vaihingen-coverage-scienceplots.png)

**Figure 2.** Scene tiles, total image footprint, and reference-scored footprint by split. Pixel totals alone overstate the difference because Potsdam has finer sampling and larger test coverage.

Validation and test references use the official no-boundary/eroded labels. The ignored shares above are boundary pixels excluded from scoring, not missing imagery. Potsdam training uses participant labels, while validation and test use eroded references. Vaihingen follows the same role pattern in the prepared bundles. The label reference therefore changes by role and should remain visible in metric reporting.

The test image footprint is about 126 ha in Potsdam and 73.06 ha in Vaihingen. This makes the Potsdam test footprint 1.73 times larger by ground area, while its scored pixel count is 5.71 times larger. The gap between those ratios is largely explained by the different GSDs.

## 4. Label composition

The class IDs are 0 impervious surfaces, 1 buildings, 2 low vegetation, 3 trees, 4 cars, 5 clutter/background, and 6 ignored boundary. All proportions below use only valid class IDs 0-5.

| Class | Potsdam train | Potsdam validation | Potsdam test | Vaihingen train | Vaihingen validation | Vaihingen test |
|---|---:|---:|---:|---:|---:|---:|
| Impervious surfaces | 28.47% | 28.32% | 31.52% | 26.84% | 31.18% | 27.65% |
| Buildings | 24.88% | 33.80% | 24.93% | 25.20% | 30.09% | 26.77% |
| Low vegetation | 24.78% | 19.26% | 20.61% | 22.20% | 16.07% | 21.19% |
| Trees | 14.64% | 14.55% | 17.09% | 23.82% | 21.40% | 22.58% |
| Cars | 1.61% | 1.55% | 1.59% | 1.06% | 1.23% | 0.97% |
| Clutter/background | 5.61% | 2.52% | 4.26% | 0.88% | 0.03% | 0.84% |

![Class proportions across all prepared splits](figures/potsdam-vaihingen-class-composition-scienceplots.png)

**Figure 3.** Composition of scored pixels by class and split. The cars and clutter shares are called out below each bar because their small segments are difficult to compare visually at this scale.

The most pronounced train-set composition difference is clutter/background: its pixel share is 6.35 times higher in Potsdam than in Vaihingen. Car support is low in both datasets. Trees are more prevalent in Vaihingen training labels, at 23.82% versus 14.64% in Potsdam. Across the class proportions, the train-to-test total variation distance is 5.54 percentage points for Potsdam and 2.38 points for Vaihingen. This measure summarizes aggregate class mix but cannot describe scene location, label boundaries, or urban morphology.

## 5. Scene-level support for minority classes

Pixel totals obscure whether a rare class is spread across scenes or concentrated in a few locations. In Potsdam, cars and clutter are present in every training scene. In Vaihingen, cars are present in all 12 training scenes, but clutter appears in only 4 of 12. The same contrast persists in evaluation: clutter appears in 14 of 14 Potsdam test scenes but only 6 of 17 Vaihingen test scenes.

| Dataset | Role | Car-bearing scenes | Clutter-bearing scenes |
|---|---|---:|---:|
| Potsdam | Train | 18 / 18 | 18 / 18 |
| Potsdam | Validation | 6 / 6 | 6 / 6 |
| Potsdam | Test | 14 / 14 | 14 / 14 |
| Vaihingen | Train | 12 / 12 | 4 / 12 |
| Vaihingen | Validation | 4 / 4 | 1 / 4 |
| Vaihingen | Test | 17 / 17 | 6 / 17 |

![Per-scene car and clutter shares in training scenes](figures/potsdam-vaihingen-rare-class-support-scienceplots.png)

**Figure 4.** Per-scene shares of valid pixels for cars and clutter/background in the training splits. The dashed line marks the median across scenes, including zero-support scenes.

The clutter/background category is heterogeneous by definition. ISPRS notes that it can include water bodies and other atypical objects such as containers, tennis courts, and swimming pools [ISPRS overview](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx). In Vaihingen, its scarcity is both class imbalance and spatial concentration. A global inverse-frequency class weight cannot create missing visual contexts; sampling and validation should preserve scene-level support.

## 6. Archived transfer and model evidence

### 6.1 Frozen-encoder cross-domain probes

An archived 19 October 2026 study froze three pretrained encoders, fitted a linear segmentation head, and scored a majority-reduced grid of approximately 5 m per cell. Cross-domain evaluation used the source normalization and head without target-label adaptation. The scores below use the eroded reference [experiment report](../experiments/potsdam-vaihingen-foundation-probe-2026-10-19.md).

| Encoder | Potsdam in-domain mIoU | Potsdam-trained to Vaihingen mIoU | Vaihingen in-domain mIoU | Vaihingen-trained to Potsdam mIoU |
|---|---:|---:|---:|---:|
| UniverSat | 22.96% | 10.11% | 29.51% | 9.41% |
| AnySat | 13.73% | 11.23% | 27.86% | 11.91% |
| TerraMind | 22.28% | 9.20% | 23.56% | 11.59% |

![Historical in-domain and cross-domain mIoU](figures/potsdam-vaihingen-domain-transfer-scienceplots.png)

**Figure 5.** In-domain held-out mIoU compared with transfer to the other city. This historical probe uses a coarse grid and is not directly comparable with full-resolution segmentation results.

All six transfer scores fall between 9.20% and 11.91% mIoU, below their in-domain scores. This is evidence of a substantial transfer gap under that probe protocol. It does not isolate the source of the gap: sensor scale, label support, neighborhood structure, and model-representation effects are entangled. The archived report explicitly requires a rerun under the current data pipeline.

### 6.2 Full-resolution joint DINOv2 run

A later historical joint DINOv2 run evaluated one selected model on both datasets. Its results provide a separate, higher-resolution reference point [final artifact review](../experiments/isprs-dinov2-joint-final-review-2026-11-01.md).

| Dataset / role | Scenes | Scored pixels | mIoU | Overall accuracy |
|---|---:|---:|---:|---:|
| Potsdam validation | 6 | 199,409,714 | 51.03% | 77.99% |
| Potsdam test | 14 | 467,862,855 | 48.65% | 74.66% |
| Vaihingen validation | 4 | 17,591,104 | 47.69% | 79.12% |
| Vaihingen test | 17 | 81,991,943 | 49.18% | 78.30% |

For that joint recipe, car IoU was 39.99% on Potsdam and 21.37% on Vaihingen; clutter IoU was 2.12% and 11.26%, respectively. These patterns do not follow class prevalence alone: Potsdam has more clutter labels but lower clutter IoU, while Vaihingen has lower car prevalence but lower car IoU. The comparison measures entire training and inference recipes, not an isolated dataset or backbone effect.

The latest audited DINOv3 benchmark in this repository is Potsdam-only, with 92.0397% OA and 81.0932% six-class mIoU on the historical 14-scene benchmark [DINOv3 final review](../experiments/potsdam-dinov3-rgbir-v2-final-review-2026-11-06.md). There is no matched DINOv3 Vaihingen evaluation in the reviewed artifacts. The DINOv3 result is not directly comparable with DINOv2 because model inputs, training recipe, refit, and inference policy differ; it also does not provide a cross-city comparison.

## 7. Discussion and recommendations

The profiles support five practical decisions for future joint experiments:

1. **Harmonize physical scale.** Declare crop width in metres and preserve a full-resolution evaluation alongside any common-ground-scale comparison.
2. **Sample by scene as well as class.** Track rare-class presence at scene and patch level. Ensure clutter appears in multiple Vaihingen scenes in training batches before tuning global class weights.
3. **Keep modality mappings explicit.** The current shared prepared view is IRRG. Treat Potsdam RGBIR/DSM as a separate modality study unless a matched Vaihingen input recipe is available and verified.
4. **Report reference masks and support.** Keep full and no-boundary scores separate. Report ignore share, scored area, per-class IoU, scene-macro metrics, and class support.
5. **Reserve spatial regions.** Current splits are scene-level rather than region-held-out. Future claims about geographic generalization should group neighboring scenes before patch extraction.

## 8. Limitations

- Train, validation, and test roles have different scene counts and physical coverage in the two datasets.
- Training uses participant labels; validation and test use eroded references. Boundary removal changes the score denominator and can change class proportions.
- Aggregate class shares do not measure within-scene arrangement, object size, radiometry, or annotation ambiguity.
- The 2026-10-19 transfer probes predate the current contract-driven pipeline and operate on a coarse majority-reduced grid.
- The DINOv2 and DINOv3 results are historical observations and are not causal comparisons of dataset identity.
- Local Vaihingen catalog records are pixel-local and do not contain CRS/affine metadata. The local catalog `license_id` remains unverified; confirm source terms before redistributing source imagery or label crops.

## 9. Methods and data audit

Class proportions were computed as counts for IDs 0-5 divided by the sum of IDs 0-5. ID 6 is the ignored no-boundary label. Split areas were estimated as pixel count multiplied by squared official GSD; these are per-tile area sums, not a GIS union of scene footprints. Scene-level rare-class presence was determined from per-scene class counts in the quality manifests. The historical transfer table was read from the archived metrics JSON; no predictions were regenerated.

The current catalog checks report `valid` for both datasets. For every train, validation, and test role, aggregate class counts equal the sum of per-scene scored pixels. The class-share total variation between train and test is computed as one half of the L1 distance between the two six-class proportion vectors.

All five visual figures were generated with Matplotlib using the **SciencePlots** `science` and `no-latex` styles. Figure 1 uses the same style for its panel typography and layout; its per-crop contrast stretch is stated in the caption.

## 10. Sources

### Official dataset references

- [ISPRS Potsdam 2D semantic labeling dataset](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)
- [ISPRS Vaihingen 2D semantic labeling dataset](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx)
- [ISPRS semantic labeling overview and class definitions](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx)

### Local data and experiment evidence

- [Potsdam train and validation quality manifest](../../data/prepared/potsdam/quality.json)
- [Potsdam test quality manifest](../../data/prepared/potsdam/post-selection-test-quality.json)
- [Vaihingen train and validation quality manifest](../../data/prepared/vaihingen/quality.json)
- [Vaihingen test quality manifest](../../data/prepared/vaihingen/post-selection-test-quality.json)
- [Potsdam and Vaihingen frozen-encoder probe](../experiments/potsdam-vaihingen-foundation-probe-2026-10-19.md)
- [Joint DINOv2 final artifact review](../experiments/isprs-dinov2-joint-final-review-2026-11-01.md)
- [Potsdam DINOv3 final artifact review](../experiments/potsdam-dinov3-rgbir-v2-final-review-2026-11-06.md)
