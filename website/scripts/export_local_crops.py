"""Export private Atlas crops from the audited paired run and held-out cache."""

import argparse
import base64
import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from export_evidence import ROOT, export

sys.path.insert(0, str(ROOT))
from deepnet.potsdam_rgbir.data import CacheBundle  # noqa: E402
from scripts.research.paired_analysis import load_prediction, verify_inputs  # noqa: E402

MODELS = ("dinov3", "pspnet")


def export_crops(kaggle_output, cache_root, runs_root, output):
    output = output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("Private raster exports must stay outside the repository")
    export()
    evidence = json.loads((ROOT / "website/src/generated/evidence.json").read_text())
    benchmark = evidence["benchmark"]
    checkpoints = {key: benchmark["models"][key]["checkpoint_sha256"] for key in MODELS}
    cache = CacheBundle.load(cache_root)
    _, paired, _, index = verify_inputs(kaggle_output, cache, runs_root)
    if paired["split_hash"] != benchmark["split_sha256"] or any(
        paired["models"][name]["checkpoint_sha256"] != checkpoints[name]
        or paired["models"][name]["archived_metrics_sha256"]
        != benchmark["models"][name]["source_metrics_sha256"]
        for name in MODELS
    ):
        raise ValueError("Paired run does not match published benchmark provenance")

    scene_cases = defaultdict(list)
    for crop in evidence["cases"]:
        scene_cases[crop["scene"]].append(crop)
    records = {
        name: {row["scene_id"]: row for row in paired["models"][name]["scenes"]}
        for name in MODELS
    }
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    with cache.access("test", auxiliary=False) as access:
        for scene, cases in scene_cases.items():
            arrays = access._arrays(scene)
            predictions = []
            for name in MODELS:
                record = records[name][scene]
                relative = "paired/" + record["prediction_path"]
                prediction = load_prediction(
                    kaggle_output / relative,
                    record["prediction_sha256"],
                    index[relative],
                )
                if prediction.shape != arrays["labels"].shape:
                    raise ValueError(f"Prediction geometry does not match {scene}")
                predictions.append(prediction)
            for crop in cases:
                row, col, height, width = (
                    crop[k] for k in ("row", "col", "height", "width")
                )
                window = np.s_[row : row + height, col : col + width]
                truth = arrays["labels"][window]
                valid = arrays["valid"][window].astype(bool) & (truth < 6)
                planes = np.stack(
                    [truth, *(prediction[window] for prediction in predictions)]
                )
                if planes.shape != (3, height, width) or np.any(planes[:, valid] >= 6):
                    raise ValueError(
                        f"Unexpected class IDs or crop geometry for {scene}"
                    )
                keys = (
                    planes[0, valid].astype(int) * 6 + planes[1, valid]
                ) * 6 + planes[2, valid]
                expected = np.zeros(216, dtype=np.int64)
                for t, d, p, count in crop["transitions"]:
                    expected[(t * 6 + d) * 6 + p] = count
                if not np.array_equal(np.bincount(keys, minlength=216), expected):
                    raise ValueError(
                        f"Published joint counts differ for {scene} at {row}, {col}"
                    )
                planes[:, ~valid] = 6
                package = {
                    "schema": 1,
                    "scene": scene,
                    "row": row,
                    "col": col,
                    "height": height,
                    "width": width,
                    "checkpoints": checkpoints,
                    "split_sha256": paired["split_hash"],
                    "label_protocol": benchmark["label_protocol"],
                    "classes": benchmark["class_names"],
                    "planes": base64.b64encode(zlib.compress(planes.tobytes())).decode(
                        "ascii"
                    ),
                }
                (output / f"potsdam-{scene}-{row}-{col}.atlas.json").write_text(
                    json.dumps(package, separators=(",", ":")) + "\n", encoding="utf-8"
                )
    print(
        f"Exported {len(evidence['cases'])} private, count-verified crop records "
        f"to {output}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("kaggle-output", "cache", "runs-root", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    export_crops(args.kaggle_output, args.cache, args.runs_root, args.output)


if __name__ == "__main__":
    main()
