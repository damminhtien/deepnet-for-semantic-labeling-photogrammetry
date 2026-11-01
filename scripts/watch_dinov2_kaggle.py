"""Monitor a Kaggle run, retrieve terminal outputs, and audit their evidence."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from deepnet.data.raster import sha256_file
from deepnet.training_utils import write_json

LOGGER = logging.getLogger(__name__)


def audit(root: Path) -> dict:
    errors = []
    manifests = list(root.rglob("run_manifest.json"))
    if len(manifests) != 1:
        return {"status": "incomplete", "errors": ["Expected one run manifest"]}
    output = manifests[0].parent
    manifest = json.loads(manifests[0].read_text())
    index = json.loads((output / "artifact_sha256_manifest.json").read_text())
    for record in index["files"]:
        path = output / record["path"]
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            errors.append(f"Artifact checksum differs: {record['path']}")
    if manifest["status"] != "complete":
        failed = manifest["status"] == "failed"
        if failed:
            errors.append("Training run failed; inspect training.log")
        return {
            "status": "failed" if failed or errors else "incomplete",
            "run_status": manifest["status"],
            "artifacts_checked": len(index["files"]),
            "errors": errors,
        }
    metrics = json.loads((output / "metrics.json").read_text())
    selection = json.loads((output / "selection.json").read_text())
    if sha256_file(output / "best.pt") != selection["checkpoint_sha256"]:
        errors.append("Selected checkpoint differs")
    if (
        metrics["selection"] != selection
        or sha256_file(output / "metrics.json") != manifest["metrics_sha256"]
    ):
        errors.append("Metrics selection/checksum differs")
    summaries = {}
    for dataset, result in metrics["test"].items():
        total = np.asarray(result["confusion_matrix"], dtype=np.int64)
        scenes = sum(
            (
                np.asarray(value["confusion_matrix"], dtype=np.int64)
                for value in result["scenes"]
            ),
            np.zeros_like(total),
        )
        if (
            not np.array_equal(total, scenes)
            or int(total.sum()) != result["pixels_scored"]
        ):
            errors.append(f"Per-scene counts differ: {dataset}")
        if result["selection"] != selection or manifest["test_evaluations"][
            dataset
        ] != {"status": "complete", "runs": 1}:
            errors.append(f"Test selection/count differs: {dataset}")
        summaries[dataset] = {
            key: result[key]
            for key in (
                "mean_iou",
                "overall_accuracy",
                "per_class_iou",
                "pixels_scored",
            )
        }
    return {
        "status": "verified" if not errors else "failed",
        "errors": errors,
        "artifacts_checked": len(index["files"]),
        "selection": selection,
        "test": summaries,
    }


def watch(kernel: str, root: Path, interval: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    while True:
        result = subprocess.run(
            ["kaggle", "kernels", "status", kernel], capture_output=True, text=True
        )
        status = result.stdout + result.stderr
        LOGGER.info("%s", status.strip())
        write_json(
            root / "watch-state.json",
            {
                "kernel": kernel,
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "status_output": status,
                "exit_code": result.returncode,
            },
        )
        with (root / "remote-log.txt").open("w") as stream:
            subprocess.run(
                ["kaggle", "kernels", "logs", kernel],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode == 0 and any(
            value in status for value in (".COMPLETE", ".ERROR", ".CANCELLED")
        ):
            download = subprocess.run(
                [
                    "kaggle",
                    "kernels",
                    "output",
                    kernel,
                    "-p",
                    str(root / "outputs"),
                    "--force",
                    "--quiet",
                ],
                check=False,
            )
            if download.returncode:
                LOGGER.error("Output download failed; retrying after the next poll")
                time.sleep(interval)
                continue
            try:
                report = audit(root / "outputs")
            except (KeyError, ValueError, OSError) as error:
                report = {"status": "failed", "errors": [str(error)]}
            write_json(root / "local-audit.json", report)
            LOGGER.info("Terminal artifacts retrieved and audited")
            return
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    watch(args.kernel, args.output, args.interval)


if __name__ == "__main__":
    main()
