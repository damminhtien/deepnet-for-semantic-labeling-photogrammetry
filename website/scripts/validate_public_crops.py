"""Validate public raster crops, PNG checksums, IDs, and exact joint counts."""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "website/public"
MANIFEST = ROOT / "website/src/generated/public-crops.json"
CLASS_COUNT = 6
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def paeth(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    distances = (
        abs(estimate - left),
        abs(estimate - above),
        abs(estimate - upper_left),
    )
    return (left, above, upper_left)[distances.index(min(distances))]


def read_png(path: Path) -> tuple[int, int, int, bytearray]:
    payload = path.read_bytes()
    if not payload.startswith(PNG_SIGNATURE):
        raise ValueError(f"Invalid PNG signature: {path}")
    cursor = len(PNG_SIGNATURE)
    width = height = bit_depth = color_type = interlace = None
    compressed = bytearray()
    while cursor < len(payload):
        size = struct.unpack_from(">I", payload, cursor)[0]
        cursor += 4
        kind = payload[cursor : cursor + 4]
        cursor += 4
        data = payload[cursor : cursor + size]
        cursor += size
        expected_crc = struct.unpack_from(">I", payload, cursor)[0]
        cursor += 4
        actual_crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError(f"Invalid PNG chunk checksum: {path}")
        if kind == b"IHDR":
            width, height, bit_depth, color_type, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", data)
            )
            if compression != 0 or filtering != 0:
                raise ValueError(f"Unsupported PNG codec: {path}")
        elif kind == b"IDAT":
            compressed.extend(data)
        elif kind == b"IEND":
            break
    if width is None or height is None or bit_depth != 8 or interlace != 0:
        raise ValueError(f"Unsupported PNG layout: {path}")
    channels = {0: 1, 2: 3}.get(color_type)
    if channels is None:
        raise ValueError(f"Expected grayscale or RGB PNG: {path}")
    row_bytes = width * channels
    decoded = zlib.decompress(compressed)
    if len(decoded) != height * (row_bytes + 1):
        raise ValueError(f"PNG raster length differs from its geometry: {path}")
    output = bytearray(height * row_bytes)
    previous = bytearray(row_bytes)
    source_offset = 0
    for row_index in range(height):
        filter_type = decoded[source_offset]
        source_offset += 1
        encoded = decoded[source_offset : source_offset + row_bytes]
        source_offset += row_bytes
        row = bytearray(row_bytes)
        for index, value in enumerate(encoded):
            left = row[index - channels] if index >= channels else 0
            above = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = above
            elif filter_type == 3:
                predictor = (left + above) // 2
            elif filter_type == 4:
                predictor = paeth(left, above, upper_left)
            else:
                raise ValueError(f"Unsupported PNG filter {filter_type}: {path}")
            row[index] = (value + predictor) & 255
        start = row_index * row_bytes
        output[start : start + row_bytes] = row
        previous = row
    return width, height, channels, output


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_public_path(relative: str) -> Path:
    path = (PUBLIC / relative).resolve()
    if PUBLIC.resolve() not in path.parents:
        raise ValueError(f"Public evidence path escapes website/public: {relative}")
    return path


def validate() -> dict[str, Any]:
    manifest = read_json(MANIFEST)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported public crop manifest schema")
    if manifest.get("dataset_id") != "isprs-potsdam":
        raise ValueError("Public crop manifest has the wrong dataset identity")
    if len(manifest.get("class_names", [])) != CLASS_COUNT:
        raise ValueError("Public crop manifest must declare six classes")
    assets = manifest.get("assets", [])
    if len(assets) != 20 or len({asset["case_id"] for asset in assets}) != 20:
        raise ValueError("Expected 20 unique curated crop records")
    seen_files: dict[str, str] = {}
    total_scored = 0
    for asset in assets:
        case_id = asset["case_id"]
        window = asset["pixel_window"]
        width, height = window["width"], window["height"]
        if (
            window["row"] < 0
            or window["col"] < 0
            or not 1 <= width <= 512
            or not 1 <= height <= 512
        ):
            raise ValueError(f"Invalid pixel window for {case_id}")
        if asset["asset_type"] != "pixel_aligned_potsdam_crop":
            raise ValueError(f"Unexpected asset type for {case_id}")
        if asset["models"] != manifest["models"]:
            raise ValueError(f"Model or checkpoint binding differs for {case_id}")
        if asset["evaluation_status"]["independent_benchmark_claim"] is not False:
            raise ValueError(f"Historical benchmark caveat missing for {case_id}")
        if asset["evaluation_status"]["training_performed"] is not False:
            raise ValueError(f"Supplemental inference status differs for {case_id}")
        if asset["evaluation_status"]["split_sha256"] != manifest["split_sha256"]:
            raise ValueError(f"Split binding differs for {case_id}")
        for digest_value in (
            asset["source_artifacts"]["rgbir"]["sha256"],
            asset["source_artifacts"]["eroded_reference"]["sha256"],
            asset["source_artifacts"]["cache_sha256"],
            *(
                record["sha256"]
                for record in asset["source_artifacts"]["paired_predictions"].values()
            ),
        ):
            if len(digest_value) != 64:
                raise ValueError(f"Malformed source digest for {case_id}")

        decoded: dict[str, tuple[int, int, int, bytearray]] = {}
        for kind in ("rgb", "infrared", "class_ids"):
            record = asset["files"][kind]
            relative = record["url"]
            path = safe_public_path(relative)
            if not path.is_file() or path.stat().st_size != record["bytes"]:
                raise ValueError(f"Missing or truncated {kind} asset for {case_id}")
            actual_sha = digest(path)
            if (
                actual_sha != record["sha256"]
                or manifest["published_files"].get(relative) != actual_sha
            ):
                raise ValueError(f"Checksum mismatch for {case_id}/{kind}")
            seen_files[relative] = actual_sha
            decoded[kind] = read_png(path)
            image_width, image_height, channels, _ = decoded[kind]
            expected_channels = 1 if kind == "infrared" else 3
            if (image_width, image_height, channels) != (
                width,
                height,
                expected_channels,
            ):
                raise ValueError(f"Unexpected {kind} raster geometry for {case_id}")

        _, _, _, class_bytes = decoded["class_ids"]
        counts = [0] * (CLASS_COUNT**3)
        scored = 0
        for offset in range(0, len(class_bytes), 3):
            truth, dino, psp = class_bytes[offset : offset + 3]
            if truth == manifest["ignore_class_id"]:
                continue
            if truth >= CLASS_COUNT or dino >= CLASS_COUNT or psp >= CLASS_COUNT:
                raise ValueError(f"Invalid class ID in {case_id}")
            counts[(truth * CLASS_COUNT + dino) * CLASS_COUNT + psp] += 1
            scored += 1
        expected = [0] * (CLASS_COUNT**3)
        for truth, dino, psp, count in asset["transitions"]:
            if not (
                0 <= truth < CLASS_COUNT
                and 0 <= dino < CLASS_COUNT
                and 0 <= psp < CLASS_COUNT
                and count > 0
            ):
                raise ValueError(f"Invalid transition row in {case_id}")
            expected[(truth * CLASS_COUNT + dino) * CLASS_COUNT + psp] = count
        if counts != expected or scored != window["valid_pixels"]:
            raise ValueError(
                f"Class-ID pixels do not reproduce exact transitions for {case_id}"
            )
        total_scored += scored

    if set(seen_files) != set(manifest["published_files"]):
        raise ValueError("Published file inventory contains missing or extra assets")
    if manifest.get("distribution_rights_status") != (
        "published_with_project_author_approval; upstream_terms_apply"
    ):
        raise ValueError("Upstream rights status is missing")
    return {
        "assets": len(assets),
        "files": len(seen_files),
        "scored_pixels": total_scored,
    }


if __name__ == "__main__":
    result = validate()
    print(
        f"Validated {result['assets']} real crops, {result['files']} assets, "
        f"and {result['scored_pixels']:,} exact scored pixels"
    )
