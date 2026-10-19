# Repository architecture

This repository contains research code and notebooks for semantic labeling of the ISPRS Potsdam imagery. `semantic_labeling/` now holds shared data and evaluation helpers, but the repository still has no package metadata or dependency lockfile.

## File map

| Path | Role |
| --- | --- |
| Insight-data-potsdam.ipynb | Data visualization and analysis notebook linked from the README |
| model_kernel/ | Model experiment notebooks |
| model_script/ | Python definitions for FCN-ResNet101, PSPNet-ResNet101, and UNet-ResNet101 variants |
| semantic_labeling/ | Shared label, preprocessing, sampling, loss, metrics, dataset, inference, and model-initialization code |
| test_kernel/ | Evaluation notebook and a compatibility entry point in test.py |
| visualize_net_structure/ | Notebooks for visualizing model structures |
| dataset_description.md | Background on the Potsdam imagery, DSM data, and benchmark labels |

The README describes the broad flow as data analysis, preprocessing, model building, experimentation, validation, and testing. The repository does not define package metadata, a command-line entry point, a dependency lockfile, or root test configuration at this snapshot.

## Model and notebook boundary

The notebooks import the shared helpers from the repository root and retain their model definitions and experiment configuration. Launch them with the repository root on Python's import path. The model scripts now import their constants and dependencies explicitly and define constructors without instantiating a model at module import time; runtime import still depends on the external PyTorch/torchvision environment.

Before changing model behavior, inspect the matching notebook and Python definition. Preserve their alignment or document which one is authoritative for the change. Use Jupyter MCP to inspect or edit notebook contents.

`test_kernel/test.py` keeps the previous evaluation helper names and delegates to `semantic_labeling/`. Shared helpers are the source of truth for labels, input scaling, patch loading, sampling windows, loss, metrics, and tiled inference.
