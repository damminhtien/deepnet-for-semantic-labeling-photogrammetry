# Seven audit repairs

1. [x] Fix count-based F1 and publish a reproducible historical metric correction.
   Validate zero-TP/empty-class cases and all corrected historical records.
2. [x] Scan required image assets, masks, finite values, and view coverage in quality
   reports. Validate missing/changed imagery and independent view masks.
3. [x] Retain per-scene confusion matrices. Extend the existing evaluation test to
   verify that their sum equals the aggregate.
4. [x] Release evaluation scratch files after each scene, including failures. Validate
   the lifecycle with two synthetic scenes.
5. [x] Clip WarpedVRT reads and pad data/masks explicitly. Validate in-bounds and edge
   windows through both raster-reader APIs.
6. [x] Replace datetime.UTC with timezone.utc. Verify package/CLI imports on Python 3.10
   where an interpreter is available, without installing project training extras.
7. [x] Correct the joint Vaihingen evaluation wording without changing split members
   or training configuration. Check protocol hashes and related documentation.

After each item, record evidence in STATE.md and commit the focused change. Finish
with core tests, Ruff/format, Graphify update, and a final diff/status review.

- [x] Core tests, Ruff/format, and code-only Graphify update.
- [x] Refresh both DVC pipelines, verify four bundles, and review the final diff.
