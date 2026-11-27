"""Freeze scene-level partitions and a machine-readable initial protocol."""
import hashlib
import json
import os
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]


def main():
    scenes = list(range(100))
    random.Random(20261021).shuffle(scenes)
    groups = {"train": sorted(scenes[:70]), "validation": sorted(scenes[70:85]), "calibration": sorted(scenes[85:]),
              "test_seen": list(range(100, 130)), "test_similar": list(range(130, 160)), "test_novel": list(range(160, 190))}
    all_scenes = [value for group in groups.values() for value in group]
    assert len(all_scenes) == 190 and len(set(all_scenes)) == 190
    payload = {"version": 1, "seed": 20261021, "unit": "scene", "groups": groups,
               "camera_primary": "realsense", "camera_secondary": "kinect",
               "camera_rule": "同一场景的全部相机、视角、退化版本严格属于同一集合",
               "pilot_train": sorted(scenes[:10]), "pilot_validation": sorted(scenes[70:75]),
               "pilot_frame_step": 16, "full_frame_step": 1,
               "status": "协议初版，尚未下载数据或核查各集合物体分布",
               "pretrained_rule": "官方预训练权重仅用于工程复现；未经训练场景核验不得作为独立校准和正式对比的训练起点"}
    content = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    path = ROOT / "data/manifests/scene_splits.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError("已存在不同数据划分，请创建新的协议版本，不覆盖原划分")
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))
    summary = {"split_sha256": hashlib.sha256(content).hexdigest(), "counts": {key: len(value) for key, value in groups.items()},
               "scene_overlap": False, "data_downloaded": False, "decision": "划分文件已生成，数据与类别覆盖核查待执行"}
    report = ROOT / "results/P1/protocol.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = report.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(report))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
