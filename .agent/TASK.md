# Goal

Rerun the assigned cloud experiments: continue PSPNet on Colab and train SAM ViT-B + LoRA + six-class decoder on Kaggle. Persist Colab logs and checkpoints to the project folder on Google Drive and copy them to local project Data, repair Kaggle source/dependency delivery, and keep the two model jobs separate.

# Acceptance

- Colab starts from the verified PSPNet epoch-2 checkpoint and exact existing data bundle; a unique run ID prevents overwriting the failed attempt.
- Colab run manifests, progress logs, latest resumable checkpoint, final model, and metrics are copied to the existing project Drive folder during the run and downloaded to local project Data with size and SHA-256 verification.
- Kaggle's existing Potsdam bundle is verified before training, the missing `semantic_labeling` dependency is attached as a private source dataset, and runtime library/GPU versions are recorded before training.
- Kaggle completes training and validation/held-out scoring, or the precise cloud/runtime blocker and last durable checkpoint are recorded.
- Kaggle artifacts remain on this Mac under `/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/`; do not save them to Drive.
- Do not claim either run complete without its final manifest and metrics. Keep all datasets, checkpoints, logs, and predictions outside Git.
