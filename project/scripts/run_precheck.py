"""Run the environment and crash-recovery checks as one pipeline stage."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    for script in ["preflight.py", "verify_resume.py"]:
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=str(ROOT))
        if result.returncode:
            sys.exit(result.returncode)
