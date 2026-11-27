"""Bounded HTTP Range ZIP reader with CRC checking and persistent range cache."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import time
import urllib.request
import zipfile
import zlib


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value) -> None:
    atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


class RangeReader(io.RawIOBase):
    """Never falls back to downloading the whole archive when Range is ignored."""

    def __init__(self, url: str, cache_dir: Path | None = None, timeout=45):
        self.url = url
        self.resolved_url = url
        self.timeout = timeout
        self.position = 0
        self.size = None
        self.etag = None
        self.network_bytes = 0
        self.cache_dir = cache_dir
        self.range(0, 1, cache=False)

    def range(self, start: int, length: int, cache=True) -> bytes:
        if length == 0:
            return b""
        if start < 0 or length < 0 or length > 64 * 1024 * 1024:
            raise ValueError("Invalid or excessive byte range")
        end = start + length - 1
        key = hashlib.sha256((self.url + str(self.size) + str(self.etag)).encode()).hexdigest()[:20]
        target = self.cache_dir / key / (str(start) + "-" + str(end) + ".bin") if self.cache_dir else None
        if cache and target and target.exists() and target.stat().st_size == length:
            return target.read_bytes()
        for attempt in range(4):
            try:
                request = urllib.request.Request(self.resolved_url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity", "User-Agent": "GraspNet-Research-Data/1.0"})
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if response.status != 206 or not match:
                        raise ValueError("Server did not honor HTTP Range; full transfer refused")
                    actual_start, actual_end, total = map(int, match.groups())
                    if (actual_start, actual_end) != (start, end):
                        raise ValueError("Incorrect Content-Range")
                    etag = response.headers.get("ETag")
                    if self.size is not None and (self.size != total or self.etag != etag):
                        raise ValueError("Remote archive changed during download")
                    data = response.read(length + 1)
                    if len(data) != length:
                        raise ValueError("Incomplete range response")
                    self.size, self.etag = total, etag
                    self.resolved_url = response.url
                    self.network_bytes += len(data)
                if cache and target:
                    atomic_write(target, data)
                return data
            except Exception:
                self.resolved_url = self.url
                if attempt == 3:
                    raise
                time.sleep(1 + attempt)
        raise RuntimeError("Unreachable")

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence == 0:
            new = offset
        elif whence == 1:
            new = self.position + offset
        elif whence == 2:
            new = self.size + offset
        else:
            raise ValueError("Invalid whence")
        if new < 0:
            raise ValueError("Negative seek")
        self.position = new
        return new

    def read(self, size=-1):
        size = min(self.size - self.position, size if size >= 0 else self.size - self.position)
        if size <= 0:
            return b""
        result = self.range(self.position, size)
        self.position += len(result)
        return result


class RemoteZip:
    def __init__(self, url: str, cache_dir: Path | None = None):
        self.reader = RangeReader(url, cache_dir)
        with zipfile.ZipFile(self.reader) as archive:
            self.entries = {item.filename: item for item in archive.infolist()}

    def read(self, name: str) -> bytes:
        info = self.entries[name]
        if info.flag_bits & 1:
            raise ValueError("Encrypted ZIP entries are unsupported")
        if info.file_size > 64 * 1024 * 1024:
            raise ValueError("Member exceeds sample size bound")
        length = min(info.compress_size + 4096, self.reader.size - info.header_offset)
        raw = self.reader.range(info.header_offset, length, cache=False)
        header = struct.unpack_from("<4s5H3I2H", raw)
        if header[0] != b"PK\x03\x04":
            raise ValueError("Invalid ZIP local header")
        header_size = 30 + header[-2] + header[-1]
        member_name = raw[30:30 + header[-2]].decode("utf-8" if info.flag_bits & 2048 else "cp437")
        if member_name != name:
            raise ValueError("ZIP local/central member mismatch")
        if header_size + info.compress_size > len(raw):
            raw = self.reader.range(info.header_offset, header_size + info.compress_size, cache=False)
        payload = raw[header_size:header_size + info.compress_size]
        if info.compress_type == zipfile.ZIP_STORED:
            result = payload
        elif info.compress_type == zipfile.ZIP_DEFLATED:
            result = zlib.decompress(payload, -15)
        else:
            raise ValueError("Unsupported ZIP compression")
        if len(result) != info.file_size or zlib.crc32(result) & 0xFFFFFFFF != info.CRC:
            raise ValueError("ZIP member size or CRC mismatch")
        return result


def safe_destination(root: Path, name: str) -> Path:
    parts = PurePosixPath(name).parts
    if ".." in parts or not parts or PurePosixPath(name).is_absolute() or "\\" in name or ":" in name:
        raise ValueError("Unsafe ZIP member path")
    target = root.joinpath(*parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("ZIP member escapes data root")
    return target
