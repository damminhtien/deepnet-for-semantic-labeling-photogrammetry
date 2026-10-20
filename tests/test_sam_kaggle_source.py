import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from model_script.train_sam_kaggle import (
    extract_source_archive,
    verify_source_package,
)


class KaggleSourceArchiveTests(unittest.TestCase):
    def test_extracts_and_verifies_source_package_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            archive_path = root / "semantic_labeling.zip"
            destination = root / "extracted"
            package_file = "semantic_labeling/__init__.py"
            content = b"# source package\n"

            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(package_file, content)

            project_root = extract_source_archive(archive_path, destination)
            record = {
                "path": package_file,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            manifest = {
                "files": [record],
                "source_sha256": hashlib.sha256(
                    json.dumps([record], sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }
            manifest_path = root / "source_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            self.assertEqual(
                verify_source_package(project_root, manifest_path),
                hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            )

    def test_rejects_archive_path_traversal(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            archive_path = root / "semantic_labeling.zip"
            destination = root / "extracted"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../outside.py", "raise SystemExit()")

            with self.assertRaisesRegex(RuntimeError, "Unsafe path"):
                extract_source_archive(archive_path, destination)

            self.assertFalse((root / "outside.py").exists())
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
