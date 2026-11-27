"""Validate the isolated vision environment on the actual GPU."""
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TORCH_HOME", str(ROOT / "runtime/cache/torch"))
os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / "runtime/cache/cuda"))


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def main():
    import numpy as np
    import torch
    import torchvision
    from torchvision.ops import nms
    report = {"date": datetime.datetime.now().astimezone().isoformat(), "python": sys.version, "executable": sys.executable,
              "base_prefix": sys.base_prefix, "isolated_prefix": sys.prefix, "platform": platform.platform(),
              "packages": {name: importlib.metadata.version(name) for name in ["torch", "torchvision", "numpy", "scipy", "Pillow", "opencv-python-headless"]},
              "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "research_grasp_pipeline_ready": False}
    if not torch.cuda.is_available():
        report["status"] = "compute_failed"
        atomic_json(ROOT / "results/P2/environment_lock.json", report)
        raise RuntimeError("CUDA unavailable in isolated environment")
    torch.manual_seed(20261021)
    np.random.seed(20261021)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    report["gpu"] = torch.cuda.get_device_name(0)
    report["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
    model = torch.nn.Sequential(torch.nn.Conv2d(4, 16, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(16, 1, 1)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(1, 4, 240, 320, device="cuda")
    target = torch.rand(1, 1, 240, 320, device="cuda")
    start = time.perf_counter()
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.], [20., 20., 30., 30.]], device="cuda")
    kept = nms(boxes, torch.tensor([0.9, 0.8, 0.7], device="cuda"), 0.5).cpu().tolist()
    report.update({"amp_forward_backward_passed": all(np.isfinite(losses)), "torchvision_cuda_nms_passed": kept == [0, 2],
                   "compute_probe_seconds": time.perf_counter() - start, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                   "status": "vision_ready" if all(np.isfinite(losses)) and kept == [0, 2] else "compute_failed",
                   "decision": "视觉环境可用于小样本GPU训练，抓取扩展和数据另行验收"})
    report["freeze"] = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()
    report["executable_sha256"] = hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
    atomic_json(ROOT / "results/P2/environment_lock.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "vision_ready" else 1


if __name__ == "__main__":
    sys.exit(main())
