# Current state

## Task

Run two independent cloud PSPNet replicas from the same epoch-2 MPS checkpoint; keep the Mac on analysis, preprocessing, augmentation, and postprocessing.

## Status

- The local scratch run is paused after epoch 2/16 at `/Users/macbook/Data/runs/pspnet-mps-20261019/last.pt`.
- Potsdam analysis and prepared labels are at `/Users/macbook/Data/runs/pspnet-preprocessed-20261019`.
- Cloud training code and a checksummed bundle builder are in `model_script/`.
- No cloud runtime has been allocated or data uploaded. The last safe quota queries found Kaggle unauthenticated and Colab awaiting manual OAuth; recheck status and quotas after sign-in. Credential clippings were not opened or used.

## Completed

- The new PSPNet matches the notebook state-dict keys and starts randomly without pretrained weights.
- The MPS training run completed epochs 1 and 2; its checkpoint contains the model, Adam state, and random states.
- Prepared the 18 training labels and 20 no-boundary evaluation labels. The split is 18/6/14.
- Sampled the five input-channel statistics and checked all train input rasters are aligned at 6000x6000.
- Cached labels as compressed `.npz` class maps; the local label cache is about 27.3 MB.
- The no-boundary `4_12` reference has palette-encoded color drift. A unique nearest-palette mapping with a 64-level per-channel bound mapped 33,463,460 pixels; the observed maximum error was 42. The audit is recorded in the preprocessing manifest. All other labels use exact palette decoding.
- Verified all four flip/mirror combinations keep image and label transforms synchronized; saved `augmentation_preview.png`.
- Verified crop-before-normalize exactly matches the old full-image pipeline on a real 256x256 patch (`array_equal=True`). A real cached dataset sample produced a finite `(5, 256, 256)` float32 input and valid integer labels.
- Loaded the epoch-2 checkpoint strictly into the cloud PSPNet and restored the Adam state on CPU.
- A checksum-current cloud bundle is at `/Users/macbook/Data/runs/pspnet-cloud-bundle-20261019` (169 files, 9.801 GiB, SHA-256 `0b3059cb164289b304e9f988de7cc4bac23519edf3471b3e6ee83024781e6e6e`). The runner verified every file against the manifest; no auth material or notebook files are present.

## Pending

1. Wait for manual Kaggle and Colab sign-in. Do not print or read credential files or auth codes.
2. Check compute quotas, then upload the bundle privately and start both runs concurrently.
3. Download logs, checkpoints, and metrics, compare both replicas, and finish the code review/Graphify/commit steps.

## Limitations

- No historical checkpoint exists, so a new scratch result cannot reproduce the historical reported accuracy exactly.
- The 16-epoch target resumes at epoch 3 on two different CUDA runtimes; the replicas are independent and may diverge because of hardware/runtime randomness.
- `/255` scaling is retained from the repository. The DSM channel includes negative values under this scale; preprocessing records the observed statistics without silently changing the model input contract.
- Held-out class distributions are not calculated or used for model selection.

## Validation so far

- `rtk ruff check` on changed model/data scripts and modules: passed.
- Python 3.14 project-environment `compileall`: passed.
- `git diff --check`: passed.
- Bundle fingerprint and all 169 file checksums: passed.
- Local crop-pipeline exact-equivalence check: passed.
- Local dataset sample smoke check: passed.
- The local notebook's saved warning now marks cells 4–5 as unexecuted training/evaluation cells; its run manifest still says `paused_for_cloud` at epoch 2.
- Cloud CUDA smoke, full training, cloud evaluation, output download, Graphify refresh, and commit: pending.
