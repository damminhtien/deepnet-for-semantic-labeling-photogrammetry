"""Check public documentation links, claim references, and evidence consistency."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(
    r"(?P<image>!)?\[(?P<alt>[^\]]*)\]\((?P<target><[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)"
)
CLAIM_RE = re.compile(r"CLAIM-\d{3}")
STATUS_RE = re.compile(
    r"^\*\*Status: (CURRENT|HISTORICAL|SUPERSEDED|CONFIGURED_NOT_RUN)\*\*"
)


def tracked(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    return result.returncode == 0


def ignored(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def check_links(markdown_files: list[Path]) -> list[str]:
    errors: list[str] = []
    for source in markdown_files:
        text = source.read_text(encoding="utf-8")
        for match in LINK_RE.finditer(text):
            target = match.group("target").strip("<>")
            if match.group("image") and not match.group("alt").strip():
                errors.append(f"{source.relative_to(ROOT)}: image has no alt text")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or target.startswith("#"):
                continue
            destination = unquote(parsed.path)
            if not destination:
                continue
            resolved = (source.parent / destination).resolve()
            if ROOT not in resolved.parents and resolved != ROOT:
                errors.append(
                    f"{source.relative_to(ROOT)}: link escapes repository: {target}"
                )
                continue
            if not resolved.exists():
                errors.append(
                    f"{source.relative_to(ROOT)}: broken local link: {target}"
                )
                continue
            if not tracked(resolved) and ignored(resolved):
                message = (
                    f"{source.relative_to(ROOT)}: link targets Git-ignored content: "
                )
                errors.append(message + target)
    return errors


def check_experiment_statuses() -> list[str]:
    errors: list[str] = []
    reports = sorted(
        path
        for path in (ROOT / "docs/experiments").glob("*.md")
        if path.name != "README.md"
    )
    if not reports:
        return ["no experiment reports found"]
    for path in reports:
        lines = path.read_text(encoding="utf-8").splitlines()
        status_lines = lines[1:4]
        if not any(STATUS_RE.match(line) for line in status_lines):
            errors.append(
                f"{path.relative_to(ROOT)}: missing status marker after title"
            )
    return errors


def check_claim_references(markdown_files: list[Path]) -> list[str]:
    import json

    registry_path = ROOT / "artifacts/public/evidence-manifest.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    valid_ids = {claim["claim_id"] for claim in registry.get("claims", [])}
    errors: list[str] = []
    used_ids: set[str] = set()
    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        for claim_id in set(CLAIM_RE.findall(text)):
            used_ids.add(claim_id)
            if claim_id not in valid_ids:
                errors.append(
                    f"{path.relative_to(ROOT)}: unknown evidence ID {claim_id}"
                )
    for claim_id in sorted(valid_ids - used_ids):
        errors.append(
            f"evidence registry: {claim_id} is not referenced by any Markdown document"
        )
    for required in ["README.md", "docs/results.md", "docs/research.md"]:
        text = (ROOT / required).read_text(encoding="utf-8")
        if not CLAIM_RE.search(text) and required != "docs/results.md":
            errors.append(
                f"{required}: headline research claims have no evidence ID references"
            )
    return errors


def main() -> int:
    markdown_files = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    errors = check_links(markdown_files)
    errors.extend(check_experiment_statuses())
    errors.extend(check_claim_references(markdown_files))

    words = re.findall(
        r"\b[\w’'-]+\b", (ROOT / "README.md").read_text(encoding="utf-8")
    )
    if not 800 <= len(words) <= 1200:
        errors.append(f"README.md: expected 800–1,200 words, found {len(words)}")

    metrics = subprocess.run(
        [sys.executable, "scripts/recompute_public_metrics.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    if metrics.returncode:
        message = metrics.stderr.strip() or metrics.stdout.strip()
        errors.append(f"public metrics check failed: {message}")

    if errors:
        print("Public documentation check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(
        "Public documentation check passed:",
        len(markdown_files),
        "Markdown files,",
        len(words),
        "README words.",
    )
    print(metrics.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
