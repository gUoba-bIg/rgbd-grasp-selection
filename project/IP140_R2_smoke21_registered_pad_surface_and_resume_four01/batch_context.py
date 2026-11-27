"""Per-process frozen slot identity. The simulator is not imported."""
import os
from pathlib import Path

VERSION = Path(__file__).resolve().parent
ROOT = VERSION.parent
SLOTS = {9: ("017", "nominal"), 10: ("021", "miss"),
         11: ("014", "miss"), 12: ("017", "miss")}
SLOT = int(os.environ["IP140_SLOT"])
OBJECT_ID, KIND = SLOTS[SLOT]
SLOT_DIR = VERSION / "slots" / f"{SLOT:02d}"
