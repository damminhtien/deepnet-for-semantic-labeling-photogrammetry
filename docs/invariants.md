# Data and experiment invariants

## Input and labels

- The README describes five input channels: IR-R-G imagery, DSM, and normalized DSM. It also describes 256 by 256 training patches.
- test_kernel/test.py sets IN_CHANNELS = 5 and defines six model labels. Its RGB palette includes index 6 for undefined pixels, while the label names call index 0 roads and the palette calls index 0 Impervious surfaces. Reconcile this mapping across data loading, loss targets, and evaluation before changing it.
- The dataset description says ground truth for some benchmark scenes remains with the organizers. Do not use withheld evaluation labels or change benchmark splits as a workaround.

## Experiment evidence

- The README reports historical accuracy figures, but this checkout does not contain a run manifest or checkpoint that establishes those figures as reproducible results. Treat them as historical claims until a specific run can be tied to its code, data split, preprocessing, configuration, checkpoint, and metric output.
- For new evaluation results, record the scene IDs and split, preprocessing, model configuration, random seed when applicable, execution environment, checkpoint, and metric output.
- Keep training, validation, and benchmark evaluation claims separate. Do not infer performance from a code change or a successful static check.

## Model and notebook changes

- Model definitions, experiment notebooks, and evaluation code overlap. Check the relevant copies before changing behavior and document any intentional divergence.
- Do not assume a model script is independently importable: current files refer to notebook-level imports/constants and create a model at module import time.
- Select validation based on the changed boundary. This repository has no root-level test configuration or dependency manifest in the current checkout; record the exact command and environment used instead of assuming a standard test command.
