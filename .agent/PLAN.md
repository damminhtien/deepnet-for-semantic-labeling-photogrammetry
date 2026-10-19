# Plan

## M1: Inventory verified evidence — complete

- Locate the saved Potsdam pilot, external RGB probes, and deeper Potsdam/Vaihingen evaluation reports and metrics.
- Record only measured results and current primary-source research. Exclude data, weights, previews, predictions, and logs.

Validation: compare tables and metadata against the saved run reports and JSON; inspect the public ISPRS and Tile2Net sources.

## M2: Write reports and provenance — complete

- Publish the frozen-encoder probe summary and machine-readable aggregate metrics.
- Publish the detailed Potsdam/Vaihingen report and its metrics artifact.
- Record official leaderboard context, a provisional model recommendation, the SAM run status, and pedestrian-path research.
- Link the reports from README and document the data boundary in architecture/invariant docs.

Validation: Python 3.14 JSON parsing; compare the reported metrics with the source JSONs; scan for raw artifact paths.

## M3: Review and publish on GitHub — complete

- Inspect status and diff; stage explicit files only.
- Run `git diff --check` and narrow JSON validation.
- Commit the report set, push local `master` including the six user-authorized existing commits, and verify remote `master` against local HEAD.
- Published commit `4da7c9e` and the six existing local commits to `master`.
- Verified `git ls-remote origin refs/heads/master` equals local HEAD `4da7c9eece479e903e2ff786b8bdfcfd4fe7ee51`.
