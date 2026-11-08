"""Build the Atlas from audited metadata only; never discover or copy rasters."""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.recompute_public_metrics import load_evidence, sha256  # noqa: E402

PUBLIC = ROOT / "artifacts/public/data"
PROFILE = (
    "docs/research/data/potsdam-vaihingen-optical-channel-histograms-2026-11-07.csv"
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def channel_histograms():
    histograms = {}
    for row in read_csv(ROOT / PROFILE):
        if row["role"] != "train":
            continue
        bands = histograms.setdefault(row["dataset"], {})
        bins = bands.setdefault(row["channel"], [None] * 256)
        dn, count = int(row["digital_number"]), int(row["pixel_count"])
        if not 0 <= dn < 256 or count < 0 or bins[dn] is not None:
            raise ValueError("Invalid optical channel histogram")
        bins[dn] = count
    if not histograms or any(
        None in bins for bands in histograms.values() for bins in bands.values()
    ):
        raise ValueError("Incomplete optical channel histograms")
    return histograms


def export():
    manifest, benchmark, _ = load_evidence()
    spectral = read_json(PUBLIC / "potsdam-rgbir-manifest.json")
    scenes = {
        name: {
            "shape": scene["shape"],
            "extent": scene["extent"],
            "crs": scene["crs"],
            "sha256": scene["sha256"],
            "historical_sha256": scene["irrg_sha256"],
            "historical_bands": scene["historical_irrg_rgbir_band_numbers"],
        }
        for name, scene in spectral["scenes"].items()
    }
    case_keys = ("case_group", "scene", "row", "col")
    cases = {
        tuple(row[key] for key in case_keys): {
            key: int(value) if value.isdecimal() else value
            for key, value in row.items()
        }
        for row in read_csv(PUBLIC / "paired-case-studies.csv")
    }
    for row in cases.values():
        row["transitions"] = []
    classes = benchmark["class_names"]
    for row in read_csv(PUBLIC / "paired-case-confusions.csv"):
        key = tuple(row[name] for name in case_keys)
        cases[key]["transitions"].append(
            [
                classes.index(row["true_class"]),
                classes.index(row["dinov3_prediction"]),
                classes.index(row["pspnet_prediction"]),
                int(row["pixels"]),
            ]
        )
    for case in cases.values():
        if sum(row[-1] for row in case["transitions"]) != case["valid_pixels"]:
            raise ValueError(f"Incomplete paired crop counts: {case['scene']}")

    study = "configs/training/potsdam-dinov3-rgbir-v2.json"
    model = "configs/models/dinov3-vitb16-rgbir-v2.json"
    data = {
        "schema": 1,
        "benchmark": benchmark,
        "registry": manifest,
        "spectral": {"bands": spectral["bands"], "scenes": scenes},
        "profile": {"path": PROFILE, "sha256": sha256(ROOT / PROFILE)},
        "histograms": channel_histograms(),
        "cases": list(cases.values()),
        "arms": read_csv(PUBLIC / "potsdam-selection-validation-arms.csv"),
        "study": read_json(ROOT / study),
        "architecture": read_json(ROOT / model),
    }
    target = ROOT / "website/src/generated/evidence.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    export()
