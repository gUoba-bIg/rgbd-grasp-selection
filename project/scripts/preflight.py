"""Collect local hardware and training readiness without changing the environment."""
from __future__ import print_function
import ctypes
import datetime
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def run(command):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
        raw = result.stdout + result.stderr
        encoding = "utf-16-le" if b"\x00" in raw[:100] else "utf-8"
        return {"returncode": result.returncode, "text": raw.decode(encoding, errors="replace").strip()}
    except Exception as exc:
        return {"returncode": -1, "text": str(exc)}


def memory_bytes():
    if os.name != "nt":
        return None
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(name, ctypes.c_ulonglong) for name in
            ["total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended"]]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.total


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def main():
    for folder in ["docs", "configs", "data/raw", "data/downloads", "data/manifests", "data/processed", "checkpoints",
                   "logs", "results/P0", "results/P1", "runtime/cache", "runtime/tmp", "third_party", "exports"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    report = {"timestamp": datetime.datetime.now().astimezone().isoformat(), "root": str(ROOT),
              "python": sys.version, "executable": sys.executable, "system": platform.platform(),
              "ram_bytes": memory_bytes(), "cpu_logical": os.cpu_count(),
              "disk_free_bytes": shutil.disk_usage(str(ROOT)).free,
              "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader,nounits"]),
              "wsl": run(["wsl.exe", "--list", "--quiet"]), "nvcc": run(["nvcc", "--version"]),
              "packages": {name: bool(importlib.util.find_spec(name)) for name in ["torch", "torchvision", "numpy", "scipy", "cv2", "open3d"]}}
    warnings = []
    try:
        import torch
        report["torch"] = {"version": torch.__version__, "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            torch.manual_seed(20261021)
            start = time.perf_counter()
            x = torch.randn(128, 128, device="cuda", requires_grad=True)
            loss = (x @ x.t()).square().mean()
            loss.backward()
            torch.cuda.synchronize()
            report["torch"].update({"device": torch.cuda.get_device_name(0), "forward_backward_finite": bool(torch.isfinite(x.grad).all().item()),
                                    "probe_seconds": time.perf_counter() - start, "peak_bytes": torch.cuda.max_memory_allocated()})
        else:
            warnings.append("未检测到可用的CUDA计算设备")
    except Exception as exc:
        report["torch_error"] = repr(exc)
        warnings.append("张量计算检查未通过")
    if sys.version_info < (3, 10):
        warnings.append("现有Python版本较旧，仅作为当前断点机制自检环境；正式基线须使用独立且锁定版本的环境")
    if report["wsl"]["returncode"] != 0 or not report["wsl"]["text"].strip():
        warnings.append("未确认可用的WSL发行版；Linux训练环境尚未就绪")
    if report["nvcc"]["returncode"] != 0:
        warnings.append("未找到CUDA编译器；驱动支持CUDA不等于已安装编译工具链")
    else:
        version_match = re.search(r"release (\d+\.\d+)", report["nvcc"]["text"])
        runtime_version = report.get("torch", {}).get("cuda_runtime")
        if version_match and runtime_version and version_match.group(1) != runtime_version:
            warnings.append("CUDA编译器{}与张量库运行时{}不同，编译扩展前必须统一并验证".format(version_match.group(1), runtime_version))
    if report["disk_free_bytes"] < 400 * 1024**3:
        warnings.append("磁盘按小规模数据试跑预算使用，全量数据解压与缓存前必须重新估算空间")
    report["warnings"] = warnings
    probe = report.get("torch", {})
    report["environment_probe_passed"] = bool(probe.get("forward_backward_finite"))
    report["research_training_ready"] = False
    report["decision"] = "完成基础检测，正式训练环境待配置" if report["environment_probe_passed"] else "基础计算检测未通过"
    atomic_json(ROOT / "results/P0/environment.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["environment_probe_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
