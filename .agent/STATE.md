# Current state

## Task

Run two different cloud jobs: Colab continues PSPNet from epoch 2 to 16; Kaggle fine-tunes SAM ViT-B with LoRA and a six-class semantic decoder. The Mac remains on analysis, preprocessing, augmentation, and postprocessing.

## Status

- Local PSPNet training is paused after epoch 2/16 at `/Users/macbook/Data/runs/pspnet-mps-20261019/last.pt`.
- Colab is assigned only to PSPNet. The active runtime is a Tesla T4. All 169 bundle files passed remote SHA-256 checks. After removing 38 non-manifest macOS `._` metadata sidecars, CUDA forward/backward smoke passed with input `[8, 5, 256, 256]`, six-class outputs, and finite loss `0.9889845848`. The trainer resumed from epoch 2 and is running toward epoch 16; no new epoch has completed yet. Current remote log: `/content/pspnet-runs/pspnet-colab-20261019.log`.
- Kaggle is assigned only to SAM. The private dataset `damminhtien/sam-vitb-lora-potsdam-20261019` is ready (3.88 GiB), and the private kernel `https://www.kaggle.com/code/damminhtien/sam-vit-b-lora-six-class-potsdam-segmentation` is queued. The last quota query reported 30 GPU hours remaining.
- Cloud code includes a SAM ViT-B wrapper with rank-4 LoRA on attention qkv/proj and a six-class decoder, a Kaggle trainer, a private bundle builder, and a private kernel stager. Local syntax/style checks passed; CUDA execution is pending.
- The Google Drive folder and small README exist, but the large bundle has not been uploaded. The resumable upload is waiting for the user's manual Drive OAuth approval in the open tab/terminal.

## Completed local preparation

- Prepared Potsdam data and labels at `/Users/macbook/Data/runs/pspnet-preprocessed-20261019`.
- Verified the 18/6/14 train/validation/held-out split and five-channel input alignment.
- Cached class-ID labels; audited palette decoding and the bounded no-boundary color remapping for tile `4_12`.
- Verified synchronized flips/mirrors and crop-before-normalize equivalence on a real patch.
- Saved the PSPNet epoch-2 model, Adam state, and random states. The SAM run starts from the separate official SAM ViT-B checkpoint, not the PSPNet checkpoint.
- The PSPNet cloud bundle is at `/Users/macbook/Data/runs/pspnet-cloud-bundle-20261019` (169 files, 9.801 GiB, SHA-256 `0b3059cb164289b304e9f988de7cc4bac23519edf3471b3e6ee83024781e6e6e`); its checksums were verified.
- The reusable archive is at `/Users/macbook/Data/runs/pspnet-colab-shared-bundle-20261019.tar` (10,524,434,944 bytes); it is staged for Drive upload. The Drive app's direct upload limit rejected files over 100 MiB, so use the separate, narrowly scoped resumable Drive API authorization rather than splitting the archive.

## SAM experiment design

- Train on the same 18 participant-labeled scenes, validate on 6 scenes, and evaluate on 14 held-out scenes.
- Use the three stored IRRG channels only; omit DSM and nDSM. Apply official SAM channel mean/std while retaining the stored IRRG order. This is a spectral-domain difference from natural RGB pretraining and the PSPNet five-channel `/255` input.
- Freeze the original SAM ViT-B image encoder weights; train rank-4 LoRA on attention qkv/proj and the new decoder for classes 0–5. Ignore label 6.
- Default run: 5 epochs, 256 random 512x512 patches per epoch, batch size 1. Evaluate validation and held-out splits after training, and write an overlay preview, metrics, logs, and per-epoch adapter/decoder checkpoints under Kaggle working output.
- Pin the official SAM source to commit `dca509fe793f601edb92606367a655c15ac00fdf`; download the ViT-B checkpoint from Meta's official URL at runtime and record its hash.

## Pending

1. Continue PSPNet epochs 3–16 on Colab and preserve checkpoints/logs after the verified bundle and passing CUDA smoke.
2. Wait for the Kaggle SAM kernel to leave the queue; verify CUDA smoke, all five epochs, held-out evaluation, and saved outputs.
3. Complete the user's Drive OAuth approval and upload the reusable PSPNet archive as one resumable file; verify Drive metadata and update the mount/extract README.
4. Download cloud checkpoints, logs, manifests, previews, and metrics under `/Users/macbook/Data`; update state, refresh Graphify if source code changes, and commit only task-owned paths.

## Limitations

- No historical PSPNet checkpoint exists, so its new scratch run cannot exactly reproduce the historical reported score.
- SAM's semantic experiment uses only three-band IRRG and SAM's pretrained normalization, while PSPNet uses five channels and `/255`; present their results separately with this caveat.
- The held-out evaluation labels are used only for final evaluation, not model selection.
- Local Python 3.14 currently has no installed `torch` or `tifffile`; model execution and CUDA validation must happen on Kaggle. Keep the Mac on data work and postprocessing.

## Validation so far

- Existing preprocessing bundle fingerprint and 169 checksums passed.
- Existing local crop-pipeline equivalence and dataset sample checks passed.
- Existing model/data scripts passed `rtk ruff check`, Python 3.14 `compileall`, and `git diff --check` before the SAM changes.
- SAM source passed `rtk ruff check`, Python 3.14 `compileall`, and `git diff --check`; CUDA execution remains pending while Kaggle is queued.
- Colab runtime reports CUDA available on Tesla T4; all 169 bundle checksums passed. The first launch was blocked by 38 `._*.npz` metadata sidecars, not missing/corrupt manifest files. The second launch removed those sidecars, passed forward/backward smoke with finite loss `0.9889845848`, and resumed from epoch 2. Epochs 3–16 remain in progress.
- Drive archive creation and local part sizing passed, but no archive bytes have been uploaded yet. Cloud result retrieval, final output verification, Graphify refresh if code changes, and commit remain pending.
