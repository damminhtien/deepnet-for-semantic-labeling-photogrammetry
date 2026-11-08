# Research claims and evidence

The [public evidence registry](../../artifacts/public/evidence-manifest.json) assigns
stable IDs to headline claims and records artifact hashes. This detailed ledger limits
each interpretation to what the archived experiments support. The primary
benchmark contains 14 Potsdam scenes that were evaluated in earlier campaigns; this is
a historical comparison, not a newly blind test set.

| Claim | Evidence | Status and scope |
| --- | --- | --- |
| The DINOv3 RGB recipe scored higher than the historical DINOv2 recipe on the recorded Potsdam cohort. | Same 14 scene IDs and 467,862,855 scored pixels; DINOv3 gains 0.5612 OA points and 1.4810 mIoU points, with 2,625,662 fewer errors. | **Supported as a historical system comparison.** Exact byte identity of the older reference rasters was not re-established, and the input recipe and tiling differ. |
| The DINOv3 backbone itself caused the 1.4810-point mIoU gain. | The campaigns also differ in channel recipe, patch size, decoder, adaptation, training schedule, and inference tiling. | **Not established.** The gain belongs to the complete evaluated recipes. |
| RGB-only DINOv3 A beat the tested bundled auxiliary recipe B. | At update 6,912 on six validation scenes, A has 80.7498% mIoU and B has 80.6097%. | **Supported for these tested configurations and this seed.** B combines IR, spectral index, DSM relief, and DSM availability. |
| Auxiliary modalities cause negative transfer. | B is lower than A by 0.1401 mIoU points; there is no single-modality ablation or enough repeated seeds to estimate a treatment effect. | **Not established.** The result does not isolate IR, index, relief, mask, fusion, or seed variation. |
| The stronger augmentation recipe improves the primary validation objective. | C is 0.2382 mIoU points below A at the same full-validation update. Car IoU rises 0.1043 points, while clutter IoU falls 1.1913 points. | **Not supported for the tested recipe and primary metric.** The class-specific trade-off is measured; its cause is not. |
| DINOv3 RGB scores better than PSPNet RGB on this cohort. | Same RGB cache, split hash, scene IDs, valid-pixel count, eroded references, 512×512 windows, 256 stride, identity inference, and batch size 4. DINOv3 is higher by 0.7563 OA points and 1.6697 mIoU points. | **Supported as a matched recipe-level comparison on the historical cohort.** It does not isolate backbone architecture. |
| DINOv3 generalizes better to an unseen city or new acquisition. | Only the previously observed Potsdam benchmark cohort is available for this comparison. | **Not established.** No independent city or acquisition was evaluated here. |
| DINOv3 and PSPNet correct the same pixels at the same rate. | Exact paired maps over 467,862,855 pixels: DINOv3-only correct 13,233,866; PSPNet-only correct 9,695,243. | **Not supported.** The paired map shows a net DINOv3 gain of 3,538,623 correct pixels, with 27,547,977 shared errors. |
| The paired errors identify why one model succeeds. | The maps establish the two predictions and eroded reference for each pixel, but include no intervention on imagery, context, or model design. | **Not established.** Class transitions and visual crops are descriptive; causal mechanisms remain untested. |
| Horizontal-flip TTA was worth its added inference cost. | Validation OA gains were 0.07858 and 0.068995 points for the two tested checkpoints, below the frozen 0.1-point threshold. | **Not supported under the predeclared selection rule.** Identity inference remained selected. |
| The DINOv3 study is cheaper than PSPNet to develop. | DINOv3 used four selection arms plus preflight, confirmation, and refit; PSPNet used one selection phase and refit. | **Depends on accounting scope.** The full experiment cost is reported separately from a winner-arm-plus-refit proxy in the quality–cost analysis. |
| These results characterize single-model systems. | Each row is one model checkpoint and uses identity inference; no ensemble is part of the reported scores. | **Supported.** The conclusions apply to the recorded non-ensemble systems. |

## Evidence sources

All aggregate and scene metrics are recomputed from archived confusion matrices. The
source checksums, manifests, phase audit files, and generated tables are recorded in
[`data/potsdam-empirical-analysis.json`](data/potsdam-empirical-analysis.json). The
confusion changes are in [`data/potsdam-confusion-transitions.csv`](data/potsdam-confusion-transitions.csv)
and [`data/potsdam-per-class-error-changes.csv`](data/potsdam-per-class-error-changes.csv).
The DINOv3 selection rows are in
[`data/potsdam-negative-results-arms.csv`](data/potsdam-negative-results-arms.csv)
and their contrasts are in
[`data/potsdam-negative-results-contrasts.csv`](data/potsdam-negative-results-contrasts.csv).
The pixel-level DINOv3/PSPNet correctness states are in
[`data/paired-prediction-summary.json`](data/paired-prediction-summary.json), with
per-class outcomes in [`data/paired-class-transitions.csv`](data/paired-class-transitions.csv)
and prediction transitions in
[`data/paired-correction-confusions.csv`](data/paired-correction-confusions.csv).

The 14-scene cohort has appeared in earlier experiments. Results here are therefore
descriptive comparisons of recorded systems and should not be presented as prospective
generalization estimates.
