import unittest

from model_script.train_sam_kaggle import (
    build_sweep_manifest,
    parse_args,
    select_sweep_winner,
    sweep_candidates,
)


class SamSweepTests(unittest.TestCase):
    def test_training_patch_does_not_change_default_eval_patch(self):
        args = parse_args(["--patch-size", "384"])

        self.assertEqual(args.patch_size, 384)
        self.assertEqual(args.eval_patch_size, 512)

    def test_three_candidates_preserve_pixel_budget_for_384_patch(self):
        candidates = sweep_candidates("retrain-test")
        by_name = {candidate["name"]: candidate for candidate in candidates}
        baseline_pixels = 256 * 512**2
        smaller_patch_pixels = by_name["candidate-b-r4-p384"]["steps_per_epoch"] * 384**2

        self.assertEqual(len(candidates), 3)
        self.assertLessEqual(abs(smaller_patch_pixels - baseline_pixels), 384**2)
        self.assertEqual(by_name["candidate-c-r8-p512"]["alpha"] / 8, 2)

    def test_sweep_manifest_tracks_all_candidate_results(self):
        candidates = sweep_candidates("retrain-test")
        manifest = build_sweep_manifest(
            parse_args(["--run-id", "retrain-test"]),
            candidates,
            {
                "bundle_sha256": "data-sha",
                "split": {
                    "train_ids": list(range(18)),
                    "validation_ids": list(range(18, 24)),
                    "held_out_test_ids": list(range(24, 38)),
                },
            },
            "Tesla T4",
        )

        self.assertEqual(manifest["candidates"], candidates)
        self.assertEqual(len(manifest["candidates"]), 3)
        self.assertEqual(manifest["candidates"][1]["patch_size"], 384)
        self.assertEqual(manifest["candidates"][1]["steps_per_epoch"], 455)
        candidates[1]["validation_miou_percent"] = 74.3
        self.assertEqual(
            manifest["candidates"][1]["validation_miou_percent"], 74.3
        )

    def test_rank4_patch512_wins_when_inside_tie_tolerance(self):
        candidates = [
            {"name": "a", "rank": 4, "patch_size": 512, "validation_miou_percent": 74.0},
            {"name": "b", "rank": 4, "patch_size": 384, "validation_miou_percent": 74.09},
            {"name": "c", "rank": 8, "patch_size": 512, "validation_miou_percent": 73.2},
        ]

        winner = select_sweep_winner(candidates)

        self.assertEqual(winner["name"], "a")

    def test_best_score_wins_when_outside_tie_tolerance(self):
        candidates = [
            {"name": "a", "rank": 4, "patch_size": 512, "validation_miou_percent": 74.0},
            {"name": "b", "rank": 4, "patch_size": 384, "validation_miou_percent": 74.11},
        ]

        self.assertEqual(select_sweep_winner(candidates)["name"], "b")


if __name__ == "__main__":
    unittest.main()
