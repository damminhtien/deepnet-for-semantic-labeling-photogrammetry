# Plan

## M1: Prepare the scratch experiment — complete

- Extract the notebook PSPNet into an importable module while preserving parameter names and random initialization.
- Create a separate local reproduction notebook and pass the MPS real-batch forward/backward smoke run.

## M2: Pause local training at the cloud handoff — complete

- Save the MPS model and Adam state after epoch 2 of 16.
- Stop the local training kernel as requested; do not continue model updates on the Mac.

## M3: Analyze and prepare Potsdam data on the Mac — complete

- Verify the 18/6/14 split, sample five-channel input statistics, and cache labels as compressed class-ID maps.
- Audit all four synchronized vertical-flip/horizontal-mirror combinations.
- Preserve the exact `/255` patch values while caching raw input rasters.
- Record the bounded color remapping required by the no-boundary `4_12` reference tile.

## M4: Build and verify the cloud bundle — complete

- Package the selected 38 scenes, prepared labels, source code, preprocessing metadata, and epoch-2 checkpoint.
- Hash every packaged file and verify all 169 inventory entries before upload.

## M5: Authenticate and check quotas — pending user sign-in

- Complete manual Colab and Kaggle OAuth in the already opened login flows.
- Check Colab compute units and Kaggle weekly GPU quota before allocating either runtime.

## M6: Train on Colab and Kaggle concurrently — pending

- Run separate 16-epoch targets from the same epoch-2 checkpoint, with CUDA smoke checks before training.
- Save resumable checkpoints after every epoch and record runtime details.

## M7: Download, compare, validate, and commit — pending

- Download both runs to `/Users/macbook/Data`, compare validation and held-out results, update state docs, update Graphify, and commit only task-owned paths.
