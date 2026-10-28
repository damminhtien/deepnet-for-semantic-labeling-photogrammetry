# Repair the seven repository-audit findings

Fix incorrect zero-true-positive F1 and correct the published historical metrics
from saved confusion counts; validate image assets in quality reports; preserve
per-scene confusion matrices; bound evaluation scratch storage to one scene; support
padded reprojected reads; retain Python 3.10 compatibility; and describe the joint
Vaihingen evaluation accurately.

Keep the fixes focused. Add regression tests only for behavior that needs them.
Preserve raw data, historical external artifacts, unrelated working-tree edits, and
the frozen train/validation/test memberships. Do not launch training or cloud jobs.
Validate each fix, refresh Graphify, and create atomic local commits.
