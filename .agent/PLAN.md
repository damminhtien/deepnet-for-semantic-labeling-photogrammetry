# Seven audit repairs

1. Fix count-based F1 and publish a reproducible historical metric correction.
   Validate zero-TP/empty-class cases and all corrected historical records.
2. Scan required image assets, masks, finite values, and view coverage in quality
   reports. Validate missing/changed imagery and independent view masks.
3. Retain per-scene confusion matrices. Extend the existing evaluation test to
   verify that their sum equals the aggregate.
4. Release evaluation scratch files after each scene, including failures. Validate
   the lifecycle with two synthetic scenes.
5. Clip WarpedVRT reads and pad data/masks explicitly. Validate in-bounds and edge
   windows through both raster-reader APIs.
6. Replace datetime.UTC with timezone.utc. Verify package/CLI imports on Python 3.10
   where an interpreter is available, without installing project training extras.
7. Correct the joint Vaihingen evaluation wording without changing split members
   or training configuration. Check protocol hashes and related documentation.

After each item, record evidence in STATE.md and commit the focused change. Finish
with core tests, Ruff/format, Graphify update, and a final diff/status review.
