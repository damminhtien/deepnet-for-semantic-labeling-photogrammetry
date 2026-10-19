# Repository architecture

This repository contains research code and notebooks for semantic labeling of the ISPRS Potsdam imagery. It is organized by activity rather than as an installable Python package.

## File map

| Path | Role |
| --- | --- |
| Insight-data-potsdam.ipynb | Data visualization and analysis notebook linked from the README |
| model_kernel/ | Model experiment notebooks |
| model_script/ | Python definitions for FCN-ResNet101, PSPNet-ResNet101, and UNet-ResNet101 variants |
| test_kernel/ | Evaluation notebook and inference/metric code in test.py |
| visualize_net_structure/ | Notebooks for visualizing model structures |
| dataset_description.md | Background on the Potsdam imagery, DSM data, and benchmark labels |

The README describes the broad flow as data analysis, preprocessing, model building, experimentation, validation, and testing. The repository does not define a centralized package, command-line entry point, dependency lockfile, or root test configuration at this snapshot.

## Model and notebook boundary

The Python model files contain notebook cell markers, rely on names such as IN_CHANNELS and N_CLASSES, and instantiate models at module scope. Treat them as notebook-coupled research artifacts unless standalone execution has been verified.

Before changing model behavior, inspect the matching notebook and Python definition. Preserve their alignment or document which one is authoritative for the change. Use Jupyter MCP to inspect or edit notebook contents.

test_kernel/test.py contains input loading, label conversion, sliding-window inference, and metric calculation. Changes to preprocessing, model outputs, or labels should be checked against this path as well as the relevant model implementation.
