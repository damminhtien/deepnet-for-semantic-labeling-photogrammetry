# Plan

## M1: Audit implementation copies

Status: complete

- Compare the evaluation script, test notebook, model notebooks, and model scripts.
- Verify dataset semantics against the checked-in description and official ISPRS material.
- Completion check: root causes and duplicated implementations are recorded.

## M2: Establish shared pipeline modules

Status: complete

- Add shared constants, label conversion, normalization, sampling, loss, metrics, and dataset loading under `semantic_labeling/`.
- Preserve existing call semantics where they are valid; reject malformed labels, image shapes, and misaligned modality lists explicitly.
- Completion check: helpers have one implementation and clear validation boundaries.

## M3: Move callers onto shared helpers and repair model paths

Status: complete

- Update the training and evaluation notebooks through Jupyter MCP and update `test_kernel/test.py`.
- Fix PSPNet auxiliary inference and invalid U-Net upsampling configuration in both source and notebook copies.
- Completion check: each caller imports the shared helpers and each source/notebook pair stays aligned.

## M4: Validate and checkpoint

Status: complete

- Inspect the complete diff, run syntax and whitespace checks, update Graphify, and commit task-owned paths.
- Do not claim model or data-path runtime validation without the missing dependencies/data/checkpoints.
- Update `.agent/STATE.md` and `.agent/DECISIONS.md` before committing.
- Completion check: exact commands and remaining limitations are recorded.
