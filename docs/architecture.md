# Repository architecture

This repository contains research code and notebooks for semantic labeling of the ISPRS Potsdam imagery. `semantic_labeling/` now holds shared data and evaluation helpers, but the repository still has no package metadata or dependency lockfile.

## File map

| Path | Role |
| --- | --- |
| Insight-data-potsdam.ipynb | Data visualization and analysis notebook linked from the README |
| model_kernel/ | Model experiment notebooks |
| model_script/ | Python definitions for FCN-ResNet101, PSPNet-ResNet101, UNet-ResNet101, and cloud experiment runners |
| semantic_labeling/ | Shared label, preprocessing, sampling, loss, metrics, dataset, inference, model-initialization, and SAM LoRA code |
| test_kernel/ | Evaluation notebook and a compatibility entry point in test.py |
| visualize_net_structure/ | Notebooks for visualizing model structures |
| dataset_description.md | Background on the Potsdam imagery, DSM data, and benchmark labels |
| docs/experiments/ and docs/research/ | Aggregate evaluation records and research notes; raw imagery and weights stay outside Git |

The README describes the broad flow as data analysis, preprocessing, model building, experimentation, validation, and testing. The repository does not define package metadata, a command-line entry point, a dependency lockfile, or root test configuration at this snapshot.

## Cloud model assignments

model_script/train_pspnet_cloud.py resumes the five-channel scratch PSPNet from the saved MPS epoch-2 checkpoint. model_script/train_sam_kaggle.py independently fine-tunes the official SAM ViT-B image encoder using LoRA and a new six-class semantic decoder. Colab owns only PSPNet; Kaggle owns only SAM. The SAM-specific dataset and kernel stagers create private artifacts under /Users/macbook/Data.

## Model and notebook boundary

The notebooks import the shared helpers from the repository root and retain their model definitions and experiment configuration. Launch them with the repository root on Python's import path. The model scripts now import their constants and dependencies explicitly and define constructors without instantiating a model at module import time; runtime import still depends on the external PyTorch/torchvision environment.

Before changing model behavior, inspect the matching notebook and Python definition. Preserve their alignment or document which one is authoritative for the change. Use Jupyter MCP to inspect or edit notebook contents.

`test_kernel/test.py` keeps the previous evaluation helper names and delegates to `semantic_labeling/`. Shared helpers are the source of truth for labels, input scaling, patch loading, sampling windows, loss, metrics, and tiled inference.

Evaluation reports are linked from the [README](../README.md). Their scope and coarse-grid limitations are recorded with each result.
