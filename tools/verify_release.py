"""Verify the source release without importing or executing scientific code."""
from pathlib import Path, PurePosixPath
import argparse
import csv
import hashlib
import json
import sys


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_path(root, relative):
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts or ":" in relative or "\\" in relative:
        raise ValueError("Unsafe manifest path: " + relative)
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path is a link or escapes the repository: " + relative)
    return path


def verify(root):
    failures = []
    seen = set()
    manifest = root / "SHA256SUMS.txt"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("Invalid digest")
        if relative in seen:
            raise ValueError("Duplicate manifest path: " + relative)
        seen.add(relative)
        path = checked_path(root, relative)
        if not path.is_file():
            failures.append({"path": relative, "reason": "missing"})
        elif sha256(path) != expected:
            failures.append({"path": relative, "reason": "hash_mismatch"})
    originals = 0
    byte_identical = 0
    with (root / "manifests/source_files.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            originals += 1
            byte_identical += row.get("original_sha256", row["sha256"]) == row["sha256"]
            path = checked_path(root, row["path"])
            if row["path"] not in seen or not path.is_file():
                failures.append({"path": row["path"], "reason": "original_missing"})
            elif path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
                failures.append({"path": row["path"], "reason": "original_identity_mismatch"})
    return {"passed": not failures, "checked_files": len(seen), "source_files": originals,
            "byte_identical_source_files": byte_identical,
            "failures": failures, "model_calls": 0, "physics_steps": 0,
            "scope": "repository file integrity only; not an experiment or cold-start test"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = verify(args.root)
    except (ValueError, OSError, KeyError) as error:
        result = {"passed": False, "error": str(error)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
