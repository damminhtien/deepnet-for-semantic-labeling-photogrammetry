# Goal

Finish the split cloud experiment: Colab continues PSPNet from the saved epoch-2 checkpoint, while Kaggle fine-tunes SAM ViT-B with LoRA and a six-class semantic decoder. The Mac remains dedicated to analysis, preprocessing, augmentation, and postprocessing.

# Cloud assignment

- **Colab:** Continue only the existing PSPNet run from epoch 2 through epoch 16. Do not train SAM in Colab.
- **Kaggle:** Train only the SAM ViT-B semantic segmentation experiment from the official pretrained checkpoint. Do not resume or train PSPNet in Kaggle.
- **Mac:** Keep the existing local PSPNet run paused after epoch 2. Prepare data, inspect results, and perform postprocessing only.

# Constraints

- Preserve the historical model_kernel/pspnet_2.ipynb and its saved outputs.
- Keep the 18/6/14 train/validation/held-out Potsdam scene split, six semantic classes (IDs 0–5), and ignore label 6.
- Colab PSPNet retains its five-channel IRRG+DSM+nDSM input and historical /255 scaling. It resumes from the epoch-2 model and Adam state.
- Kaggle SAM starts from Meta's official SAM ViT-B checkpoint. Freeze the original image encoder weights; train rank-4 LoRA updates on attention qkv and proj, plus a custom six-class decoder.
- Kaggle SAM uses the stored three-channel IRRG imagery only. It omits DSM/nDSM and uses SAM's pretrained per-channel mean/std normalization on the IRRG values. IRRG is not natural RGB; preserve and report this domain difference.
- Kaggle trains on the 18 participant-labeled scenes, validates on the six validation scenes, and evaluates once on the 14 held-out scenes. Label 6 is ignored in loss and metrics.
- Keep the Kaggle dataset and kernel private. Never publish ISPRS data or upload credentials, tokens, auth files, or notebook secrets.
- Upload the checksummed PSPNet cloud bundle to a private Google Drive folder so multiple Colab notebooks can mount and reuse it. Include a short mount/extract guide; keep access private to the user's account.
- Keep checkpoints, logs, manifests, prepared data, and cloud artifacts under /Users/macbook/Data, outside Git.
- Preserve unrelated workstation state and stage explicit task-owned paths only.
- No historical PSPNet checkpoint exists. Report its result as a new scratch run, not an exact reproduction of the historical score.
- Report SAM and PSPNet metrics separately and state their input and initialization differences.

# Acceptance

- Existing local preprocessing and augmentation checks remain complete, with bundle inputs verified by SHA-256.
- Upload the PSPNet bundle archive to private Drive and verify its metadata; document how separate Colab notebooks mount and extract the same archive.
- Colab passes a real CUDA forward/backward smoke run and completes PSPNet epochs 3–16.
- Kaggle passes a real CUDA forward/backward smoke run and completes the configured five-epoch SAM LoRA run (256 random 512x512 patches per epoch by default).
- Save resumable SAM adapter/decoder checkpoints after every epoch, then produce validation and held-out overall accuracy and mean IoU, preview overlays, logs, and runtime provenance.
- Download Colab and Kaggle outputs under /Users/macbook/Data and report measured results without conflating the architectures.
- Update Graphify, run focused syntax/style/whitespace validation, and commit only task-owned paths.
