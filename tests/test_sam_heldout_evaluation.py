import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from model_script.train_sam_kaggle import evaluate_selected_held_out


class SelectedHeldOutEvaluationTests(unittest.TestCase):
    def test_validation_selected_model_is_evaluated_once(self):
        confusion = [[1, 0], [0, 1]]
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            config_sha = "training-config-sha"
            checkpoint = {
                "training_config_sha256": config_sha,
                "sam_checkpoint_sha256": "sam-sha",
                "dataset_bundle_sha256": "data-sha",
                "trainable_state_dict": {},
                "best_validation_metrics": {"confusion_matrix": confusion},
            }
            torch.save(checkpoint, output_dir / "best.pt")
            (output_dir / "last.pt").write_bytes(b"last checkpoint")
            validation_metrics = {
                "confusion_matrix": confusion,
                "mean_iou_percent": 75.0,
            }
            (output_dir / "metrics.json").write_text(
                json.dumps(
                    {"validation": validation_metrics, "held_out_test": None}
                ),
                encoding="utf-8",
            )
            (output_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "validation_complete",
                        "held_out_evaluation_count": 0,
                        "sam_checkpoint_sha256": "sam-sha",
                        "dataset_bundle_sha256": "data-sha",
                        "lora": {"rank": 4, "alpha": 8},
                        "training_config": {"eval_patch_size": 512},
                        "training_config_sha256": config_sha,
                    }
                ),
                encoding="utf-8",
            )

            args = SimpleNamespace(rank=4, alpha=8, eval_patch_size=512)
            held_out_metrics = {"mean_iou_percent": 70.0}
            with (
                patch(
                    "model_script.train_sam_kaggle.restore_trainable_state"
                ) as restore,
                patch(
                    "model_script.train_sam_kaggle.evaluate_split",
                    return_value=held_out_metrics,
                ) as evaluate,
            ):
                evaluate_selected_held_out(
                    args,
                    output_dir,
                    Path(temporary_dir),
                    ["heldout-tile"],
                    object(),
                    "sam-sha",
                    {"bundle_sha256": "data-sha"},
                )
                self.assertEqual(evaluate.call_count, 1)
                self.assertEqual(restore.call_count, 1)

                with self.assertRaisesRegex(RuntimeError, "validation_complete"):
                    evaluate_selected_held_out(
                        args,
                        output_dir,
                        Path(temporary_dir),
                        ["heldout-tile"],
                        object(),
                        "sam-sha",
                        {"bundle_sha256": "data-sha"},
                    )
                self.assertEqual(evaluate.call_count, 1)

            metrics = json.loads((output_dir / "metrics.json").read_text())
            manifest = json.loads((output_dir / "run_manifest.json").read_text())
            self.assertEqual(metrics["held_out_evaluation_count"], 1)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["held_out_evaluation_count"], 1)


if __name__ == "__main__":
    unittest.main()
