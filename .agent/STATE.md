# Current state

## Task

Run two different cloud jobs: Colab continues PSPNet from epoch 2 to 16; Kaggle fine-tunes SAM ViT-B with LoRA and a six-class semantic decoder. The Mac remains on analysis, preprocessing, augmentation, and postprocessing.

## Status

- Local PSPNet training is paused after epoch 2/16 at `/Users/macbook/Data/runs/pspnet-mps-20261019/last.pt`.
- The original Colab session `pspnet-colab-20261019` is no longer active: `colab sessions` reported no active sessions and `status` returned not found. Its last known log reached epoch 3 batch 200/1250; no epoch 3 completion was verified. A fresh T4 session, `pspnet-colab-20261019-retry`, is available and reports Python 3.13.15, Torch 2.11.0+cu130, and CUDA on Tesla T4. Its training has not restarted because Colab CLI upload returned HTTP 500 for both the 10.5 GB archive and one 95 MiB part. Mount the Drive folder in the open Colab tab before rebuilding the archive and resuming from epoch 2.
- Kaggle is assigned only to SAM. The private dataset `damminhtien/sam-vitb-lora-potsdam-20261019` is ready (3.88 GiB), and the private kernel `https://www.kaggle.com/code/damminhtien/sam-vit-b-lora-six-class-potsdam-segmentation` is queued. The last quota query reported 30 GPU hours remaining.
- Cloud code includes a SAM ViT-B wrapper with rank-4 LoRA on attention qkv/proj and a six-class decoder, a Kaggle trainer, a private bundle builder, and a private kernel stager. Local syntax/style checks passed; CUDA execution is pending.
- The private Google Drive folder `https://drive.google.com/drive/folders/1Da5gI-FLnKsTRLxN0-xo_vPd13YVtZqn` now contains all 106 archive parts, `parts_manifest.json`, and the updated README (108 items total). Drive readback found no missing/duplicate parts or wrong sizes; total part size is 10,524,434,944 bytes. The remote manifest bytes match the local manifest exactly, and the README readback contains the Colab mount, verification, and extraction instructions. The folder remains private.
- Google OAuth was completed by the user. A direct Drive API request made with local gcloud ADC returned `403 ACCESS_TOKEN_SCOPE_INSUFFICIENT`; this was an ADC scope mismatch, not a missing user login. The Drive upload used the already-authorized connected Drive integration and was read back successfully. For future Drive file operations, use that integration; do not request another login because of an ADC scope error. Colab's `drive.mount()` is a separate in-runtime OAuth flow.

## Completed local preparation

- Prepared Potsdam data and labels at `/Users/macbook/Data/runs/pspnet-preprocessed-20261019`.
- Verified the 18/6/14 train/validation/held-out split and five-channel input alignment.
- Cached class-ID labels; audited palette decoding and the bounded no-boundary color remapping for tile `4_12`.
- Verified synchronized flips/mirrors and crop-before-normalize equivalence on a real patch.
- Saved the PSPNet epoch-2 model, Adam state, and random states. The SAM run starts from the separate official SAM ViT-B checkpoint, not the PSPNet checkpoint.
- The PSPNet cloud bundle is at `/Users/macbook/Data/runs/pspnet-cloud-bundle-20261019` (169 files, 9.801 GiB, SHA-256 `0b3059cb164289b304e9f988de7cc4bac23519edf3471b3e6ee83024781e6e6e`); its checksums were verified.
- The reusable archive is at `/Users/macbook/Data/runs/pspnet-colab-shared-bundle-20261019.tar` (10,524,434,944 bytes; SHA-256 `b0bc76f02ad827795e1eaa1fa50f4d75f1069461cc68c4336a60c307de60f88f`). Because the Drive connector caps each uploaded file at 100 MiB, its bytes are stored as 106 numbered 95 MiB parts under `/Users/macbook/Data/runs/pspnet-drive-parts-95m-20261019`, with per-part hashes in `parts_manifest.json`. The private Drive README includes the Colab mount, reconstruction, hash verification, and extraction steps.

## SAM experiment design

- Train on the same 18 participant-labeled scenes, validate on 6 scenes, and evaluate on 14 held-out scenes.
- Use the three stored IRRG channels only; omit DSM and nDSM. Apply official SAM channel mean/std while retaining the stored IRRG order. This is a spectral-domain difference from natural RGB pretraining and the PSPNet five-channel `/255` input.
- Freeze the original SAM ViT-B image encoder weights; train rank-4 LoRA on attention qkv/proj and the new decoder for classes 0–5. Ignore label 6.
- Default run: 5 epochs, 256 random 512x512 patches per epoch, batch size 1. Evaluate validation and held-out splits after training, and write an overlay preview, metrics, logs, and per-epoch adapter/decoder checkpoints under Kaggle working output.
- Pin the official SAM source to commit `dca509fe793f601edb92606367a655c15ac00fdf`; download the ViT-B checkpoint from Meta's official URL at runtime and record its hash.

## Pending

1. In the open Colab tab, mount the private Drive folder with `from google.colab import drive; drive.mount('/content/drive')`. Then reconstruct and verify the archive from its parts, extract to `/content/pspnet-bundle`, and resume PSPNet from epoch 2 on the fresh T4. Preserve checkpoints/logs.
2. Wait for the Kaggle SAM kernel to leave the queue; verify CUDA smoke, all five epochs, held-out evaluation, and saved outputs.
3. Download cloud checkpoints, logs, manifests, previews, and metrics under `/Users/macbook/Data`; update state, refresh Graphify if source code changes, and commit only task-owned paths.

## Limitations

- No historical PSPNet checkpoint exists, so its new scratch run cannot exactly reproduce the historical reported score.
- SAM's semantic experiment uses only three-band IRRG and SAM's pretrained normalization, while PSPNet uses five channels and `/255`; present their results separately with this caveat.
- The held-out evaluation labels are used only for final evaluation, not model selection.
- Local Python 3.14.8 now has `torch`, `torchvision`, `numpy`, `scikit-image`, `scikit-learn`, `matplotlib`, `tqdm`, and `tifffile` installed in the user site. Imports were verified; MPS is available and CUDA is unavailable. Keep the Mac on data work and postprocessing; CUDA experiment validation remains on Kaggle/Colab.

## Validation so far

- Existing preprocessing bundle fingerprint and 169 checksums passed.
- Existing local crop-pipeline equivalence and dataset sample checks passed.
- Existing model/data scripts passed `rtk ruff check`, Python 3.14 `compileall`, and `git diff --check` before the SAM changes.
- SAM source passed `rtk ruff check`, Python 3.14 `compileall`, and `git diff --check`; CUDA execution remains pending while Kaggle is queued.
- Historical Colab checks passed before its session disappeared: all 169 bundle checksums, cleanup of 38 non-manifest `._*.npz` sidecars, and a CUDA forward/backward smoke with finite loss `0.9889845848`. The last known original-run log reached epoch 3 batch 200/1250, with no epoch completion verified. The replacement T4 reports CUDA availability, but its model smoke and training have not yet run.
- Drive readback verified 106 numbered parts with expected sizes and 10,524,434,944 total bytes. The remote `parts_manifest.json` matches the local bytes exactly; the README was fetched and checked for the Colab mount/hash/extract instructions. Cloud output retrieval, final result verification, and remaining training are pending.
- Auth-routing fix is recorded in `docs/AGENT_WORKFLOW.md`: use the authorized Drive integration for Drive files, treat gcloud ADC as scope-independent, and use `drive.mount()` only inside Colab. This prevents a local ADC scope error from triggering another user login request.
