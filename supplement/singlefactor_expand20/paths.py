"""Independent output with read-only bindings to the published V1 sampling and physics tools."""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
PRIOR = ROOT / "outputs/V1_supplement_robustness_admission01"
SOURCE = ROOT / "worktrees/V1_supplement_robustness_admission01"
PREVIOUS_CODE = SOURCE / "supplement"
sys.path.insert(0, str(PREVIOUS_CODE))
import common_v1 as v1
from common_v1 import COMMIT, TAG, ENGINE, C, METHODS, PL, R4, R6NAME, R6FIRST
from common_v1 import need, read, sha, utc

STUDY = OUT / "study"
FACTORS = ("depth_only", "shift_only", "missing_only", "block_only")
OLD_ZIP_SHA = "65b11ddf8ca2cb21b3f6dc0c6c08d5b3fe4ffcaf540c00151278c4b326b8b04a"
OLD_LOCAL_SHA = "e645ec66706161dbfd9539977a43432699afe4bd5bf631e0cdc2d9d49da072f9"
FLOOR = 80 * 1024**3

def exclusive_bytes(path, data):
    path = Path(path)
    need(path.resolve().is_relative_to(OUT) and path.resolve() != OUT, "output escapes new version")
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    return hashlib.sha256(data).hexdigest()

def write(relative, obj):
    data = (json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
    return exclusive_bytes(OUT / relative, data)

def original_manifest():
    path = PRIOR / "delivery/SHA256SUMS_local.txt"
    need(sha(path) == OLD_LOCAL_SHA, "previous evidence index hash mismatch")
    lines = path.read_text("utf-8-sig").splitlines()
    items = {name:digest for digest,name in (line.split("  ",1) for line in lines)}
    need(len(items) == len(lines) == 49496, "previous index membership mismatch")
    return items

def verify_old(rel, manifest):
    path = PRIOR / rel
    need(rel in manifest and sha(path) == manifest[rel], "historical source mismatch: "+rel)
    return manifest[rel]

def disk():
    record = v1.disk()
    record["floor_bytes"] = FLOOR
    record["above_floor"] = record["free_bytes"] >= FLOOR
    return record
