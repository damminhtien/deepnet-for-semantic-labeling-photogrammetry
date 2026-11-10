"""Publish an allowlist of archived paired figures with audited panel geometry."""

import json
import shutil
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.recompute_public_metrics import sha256  # noqa: E402

FIGURES = (
    "dinov3_only_correct",
    "pspnet_only_correct",
    "shared_errors",
    "car_clutter_disagreement",
    "vegetation_disagreement",
)
LAYERS = ("rgb", "reference", "dinov3", "pspnet", "paired_outcome")
ROW_EDGES = ((177, 589), (604, 1016), (1032, 1444), (1459, 1871))
LAYOUTS = {
    2948: {
        "height": 1973,
        "column_edges": (
            (79, 492),
            (691, 1104),
            (1303, 1716),
            (1915, 2328),
            (2527, 2940),
        ),
    },
    3057: {
        "height": 1973,
        "column_edges": (
            (160, 573),
            (759, 1171),
            (1358, 1770),
            (1957, 2369),
            (2555, 2968),
        ),
    },
}
RIGHTS_NOTE = (
    "The project owner confirmed authorization to publish this curated "
    "research figure. It contains rendered ISPRS imagery and class maps and "
    "links to the dataset source. "
    "This figure is a selected derivative, not a copy of the source archive."
)
RIGHTS_REVIEW_URL = (
    "https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/"
    "2d-sem-label-potsdam.aspx"
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Expected a PNG figure: {path}")
    return struct.unpack(">II", header[16:24])


def panel_boxes(width: int) -> dict[str, dict[str, int]]:
    layout = LAYOUTS.get(width)
    if layout is None:
        raise ValueError(f"No audited panel layout for {width}px figure")
    boxes = {}
    for layer, (left, right) in zip(LAYERS, layout["column_edges"], strict=True):
        boxes[layer] = {
            "x": left + 1,
            "width": right - left - 1,
        }
    return boxes


def crop_id(case: dict) -> str:
    return f"{case['scene']}-{case['row']}-{case['col']}"


def export() -> dict:
    evidence_path = ROOT / "website/src/generated/evidence.json"
    if not evidence_path.exists():
        raise FileNotFoundError("Run export_evidence.py before exporting visual assets")
    evidence = read_json(evidence_path)
    benchmark = evidence["benchmark"]
    source_dir = ROOT / "docs/research/figures"
    public_dir = ROOT / "website/public/evidence/paired"
    public_dir.mkdir(parents=True, exist_ok=True)
    source_digests = {
        "case_studies": sha256(ROOT / "artifacts/public/data/paired-case-studies.csv"),
        "case_confusions": sha256(
            ROOT / "artifacts/public/data/paired-case-confusions.csv"
        ),
    }
    assets = []
    case_studies = []

    for group in FIGURES:
        filename = f"paired-cases-{group}.png"
        source = source_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing published paired figure: {source}")
        width, height = png_size(source)
        layout = LAYOUTS.get(width)
        if layout is None or height != layout["height"]:
            raise ValueError(
                f"Figure has no verified panel layout: {filename} ({width}×{height})"
            )
        digest = sha256(source)
        destination = public_dir / filename
        shutil.copyfile(source, destination)
        if sha256(destination) != digest:
            raise ValueError(f"Published copy checksum mismatch: {filename}")

        group_cases = [
            case for case in evidence["cases"] if case["case_group"] == group
        ]
        if len(group_cases) != 4:
            raise ValueError(
                "Expected four ordered case rows in "
                f"{filename}, found {len(group_cases)}"
            )
        asset_cases = []
        for row_index, case in enumerate(group_cases):
            boxes = panel_boxes(width)
            top, bottom = ROW_EDGES[row_index]
            for box in boxes.values():
                box["y"] = top + 1
                box["height"] = bottom - top - 1
            record = {
                "case_id": crop_id(case),
                "case_group": group,
                "case_group_title": case["case_group_title"],
                "row_index": row_index,
                "dataset_id": benchmark["dataset_id"],
                "scene_id": case["scene"],
                "pixel_window": {
                    "row": case["row"],
                    "col": case["col"],
                    "height": case["height"],
                    "width": case["width"],
                    "valid_pixels": case["valid_pixels"],
                },
                "ground_sampling_distance_m": 0.05,
                "channel_mapping": ["red", "green", "blue"],
                "label_protocol": benchmark["label_protocol"],
                "class_names": benchmark["class_names"],
                "models": {
                    name: {
                        "model_id": benchmark["models"][name]["experiment_id"],
                        "checkpoint_sha256": benchmark["models"][name][
                            "checkpoint_sha256"
                        ],
                        "preprocessing_protocol_sha256": benchmark["models"][name][
                            "protocol_hash"
                        ],
                    }
                    for name in ("dinov3", "pspnet")
                },
                "transitions": case["transitions"],
                "panels": boxes,
                "figure_asset_id": group,
            }
            case_studies.append(record)
            asset_cases.append(record["case_id"])

        assets.append(
            {
                "asset_id": group,
                "asset_type": "archived_paired_case_figure",
                "file": filename,
                "mime_type": "image/png",
                "width_px": width,
                "height_px": height,
                "scene_ids": sorted({case["scene"] for case in group_cases}),
                "case_ids": asset_cases,
                "source_artifact_path": f"docs/research/figures/{filename}",
                "source_artifact_sha256": digest,
                "published_asset_path": f"evidence/paired/{filename}",
                "published_asset_sha256": sha256(destination),
                "case_metadata_sha256": source_digests["case_studies"],
                "joint_count_metadata_sha256": source_digests["case_confusions"],
                "distribution_rights_status": (
                    "project_owner_authorized_curated_derivative"
                ),
                "rights_note": RIGHTS_NOTE,
                "rights_review_url": RIGHTS_REVIEW_URL,
            }
        )

    manifest = {
        "schema_version": 1,
        "dataset_id": benchmark["dataset_id"],
        "class_names": benchmark["class_names"],
        "assets": assets,
        "case_studies": case_studies,
    }
    public_manifest = public_dir / "manifest.json"
    public_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    generated = ROOT / "website/src/generated/visual-evidence.json"
    generated.write_text(
        json.dumps(manifest, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    manifest = export()
    print(
        f"Exported {len(manifest['assets'])} figures and "
        f"{len(manifest['case_studies'])} panel records"
    )
