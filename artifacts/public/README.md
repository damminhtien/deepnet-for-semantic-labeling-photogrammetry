# Public evidence bundle

This directory contains the small metadata and confusion-matrix artifacts needed to
recompute reported benchmark metrics. It excludes raster imagery, labels, prediction
maps, model weights, caches, and Kaggle outputs.

Run the independent metric check from the repository root:

```bash
python3 scripts/recompute_public_metrics.py --check
```

To regenerate the numerical section in [`docs/results.md`](../../docs/results.md):

```bash
python3 scripts/recompute_public_metrics.py --write
```

[`evidence-manifest.json`](evidence-manifest.json) maps claim IDs to experiments,
configuration references, split/protocol details, and SHA-256 digests. The metrics file
contains each model's aggregate and 14 per-scene 6×6 confusion matrices. It includes
source artifact digests so the exported measurements can be traced to the archived run
outputs available to the authors.

The files under `data/` include metadata snapshots and LF-normalized analysis tables.
They contain no image or label pixels. The analysis tables preserve the source values;
their normalization makes hashes stable across Git checkouts with different line-ending
settings. The source data and model terms remain separate; this repository does not
grant rights to redistribute the ISPRS dataset or pretrained weights. The repository
currently has no code license declaration, so reuse rights for code have not yet been
granted here.
