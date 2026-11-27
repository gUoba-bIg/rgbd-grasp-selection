"""Create the isolated Windows vision environment and retain resumable wheels."""
import argparse
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(value for key, value in attrs if key == "href")


def download_wheel(package, version):
    index = "https://download.pytorch.org/whl/cu126/{}/".format(package)
    with urllib.request.urlopen(index, timeout=45) as response:
        parser = Links()
        parser.feed(response.read().decode("utf-8"))
    expected = "{}-{}+cu126-cp312-cp312-win_amd64.whl".format(package, version)
    matches = [unescape(link) for link in parser.links if urllib.parse.unquote(urllib.parse.urlsplit(link).path).endswith("/" + expected)]
    if len(matches) != 1:
        raise RuntimeError("Expected one official wheel, found {} for {}".format(len(matches), expected))
    source = urllib.parse.urljoin(index, matches[0])
    digest = urllib.parse.parse_qs(urllib.parse.urlsplit(source).fragment).get("sha256", [None])[0]
    if not digest or not re.fullmatch("[a-f0-9]{64}", digest):
        raise RuntimeError("Official wheel has no SHA256 digest")
    url = urllib.parse.urldefrag(source)[0]
    target = ROOT / "runtime" / "wheels" / expected
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) != digest:
            raise RuntimeError("Existing wheel checksum differs: " + str(target))
        print("Verified cached wheel: " + expected, flush=True)
        return target
    part = target.with_suffix(".whl.partial")
    for attempt in range(6):
        offset = part.stat().st_size if part.exists() else 0
        # A shutdown may leave every byte durable before the final rename.
        # Verify against the official digest before requesting an EOF range.
        if offset and sha256(part) == digest:
            os.replace(part, target)
            print("Recovered verified partial wheel: " + expected, flush=True)
            return target
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = "bytes={}-".format(offset)
        try:
            response = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
            if offset and response.status != 206:
                response.close()
                raise RuntimeError("Server did not honor resume range; partial file preserved")
            if offset and not response.headers.get("Content-Range", "").startswith("bytes {}-".format(offset)):
                response.close()
                raise RuntimeError("Resume range mismatch")
            size = int(response.headers.get("Content-Length", "0")) + offset
            if shutil.disk_usage(ROOT).free < (60 * 1024**3 + size - offset):
                response.close()
                raise RuntimeError("Insufficient disk reserve")
            atomic_json(target.with_suffix(".source.json"), {"url": url, "sha256": digest, "size_bytes": size, "index": index})
            previous_print = time.monotonic()
            count = offset
            with response, part.open("ab" if offset else "wb") as stream:
                while True:
                    block = response.read(4 * 1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
                    count += len(block)
                    if time.monotonic() - previous_print > 15:
                        stream.flush()
                        os.fsync(stream.fileno())
                        print("{}: {:.1f}/{:.1f} MiB".format(package, count / 1024**2, size / 1024**2), flush=True)
                        previous_print = time.monotonic()
                stream.flush()
                os.fsync(stream.fileno())
            if size and count != size:
                raise OSError("Incomplete HTTP response")
            if sha256(part) != digest:
                raise RuntimeError("Downloaded wheel checksum mismatch; partial retained for inspection")
            os.replace(part, target)
            print("Downloaded and verified: " + expected, flush=True)
            return target
        except (OSError, TimeoutError) as exc:
            print("Download attempt {}: {}".format(attempt + 1, exc), flush=True)
            if attempt == 5:
                raise
    raise RuntimeError("Download did not complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    env_root = ROOT / "runtime/envs/vision"
    interpreter = env_root / "Scripts/python.exe"
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Run bootstrap with Python 3.12")
    if not interpreter.exists():
        subprocess.check_call([sys.executable, "-m", "venv", str(env_root)])
    environment = dict(os.environ)
    for name, relative in {"PIP_CACHE_DIR": "runtime/cache/pip", "TEMP": "runtime/tmp", "TMP": "runtime/tmp", "TORCH_HOME": "runtime/cache/torch", "CUDA_CACHE_PATH": "runtime/cache/cuda"}.items():
        path = ROOT / relative
        path.mkdir(parents=True, exist_ok=True)
        environment[name] = str(path)
    environment.update(PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    log = ROOT / "logs/P2/environment_setup.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    wheels = [download_wheel("torch", "2.6.0"), download_wheel("torchvision", "0.21.0")]
    if args.download_only:
        return
    commands = [
        [str(interpreter), "-m", "pip", "install", "--disable-pip-version-check", "--timeout", "60", "--retries", "5", "pip==25.3"],
        [str(interpreter), "-m", "pip", "install", "--disable-pip-version-check", "--timeout", "60", "--retries", "5", *map(str, wheels), "numpy==1.26.4", "scipy==1.14.1", "Pillow==11.3.0", "opencv-python-headless==4.10.0.84", "requests==2.32.5", "tqdm==4.67.1"]
    ]
    with log.open("a", encoding="utf-8") as stream:
        for command in commands:
            print("Installing isolated dependencies; log: " + str(log), flush=True)
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            stream.flush()
            if result.returncode:
                raise RuntimeError("Dependency installation failed; see " + str(log))
    freeze = subprocess.check_output([str(interpreter), "-m", "pip", "freeze"], env=environment, text=True)
    atomic_json(ROOT / "results/P2/vision_packages.json", {"interpreter": str(interpreter), "bootstrap_python": sys.executable,
        "packages": freeze.splitlines(), "wheel_sha256": {p.name: sha256(p) for p in wheels}, "status": "installed_pending_compute_validation"})
    print("Isolated dependencies installed", flush=True)


if __name__ == "__main__":
    main()
