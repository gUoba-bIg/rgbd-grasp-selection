"""Run one real RGB-D frame through the pretrained candidate generator."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / 'runtime/build/graspnet'
    frame = root / 'data/raw/graspnet/scenes/scene_0003/realsense'
    checkpoint_path = root / 'checkpoints/pretrained/checkpoint-rs.tar'
    output = root / 'results/P3/inference_smoke.json'
    candidates_path = root / 'results/P3/scene_0003_realsense_0000_candidates.npy'
    inputs = {'rgb': frame / 'rgb/0000.png', 'depth': frame / 'depth/0000.png', 'meta': frame / 'meta/0000.mat', 'checkpoint': checkpoint_path}
    file_hashes = {name: digest(path) for name, path in inputs.items()}
    if file_hashes['checkpoint'] != '60680087c61cba2b6791614fef1519071e294f6dcaf99b3f581bb95f7c51a868':
        raise ValueError('Checkpoint differs from the pinned public mirror.')
    protocol = {
        'scene': 'scene_0003', 'camera': 'realsense', 'frame': '0000',
        'seed': 20261021, 'num_points': 20000, 'maximum_depth_m': 2.0,
        'crop': 'finite depth with 0 < depth_m <= 2.0; no spatial ROI',
        'uses_ground_truth_labels': False,
        'metadata_fields_read': ['intrinsic_matrix', 'factor_depth'],
        'model_input': 'XYZ only, input_feature_dim=0; RGB decoded for input verification',
        'collision_detection': False, 'nms': False,
        'checkpoint_source': 'public mirror dgrachev/a2_pretrained revision 7856e4e550052131b6ac5c0adb6564b04433ad9b',
        'checkpoint_official_byte_identity_verified': False,
        'upstream_commit': '280c215129f759ed8649cb4e89fc5dfee55f4f80',
        'compatibility_patch_sha256': digest(root / 'third_party/patches/graspnet_windows.patch'),
        'script_sha256': digest(__file__), 'input_sha256': file_hashes
    }
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if output.exists() and candidates_path.exists():
        prior = json.loads(output.read_text(encoding='utf-8'))
        if prior.get('status') == 'passed' and prior.get('fingerprint') == fingerprint and prior.get('candidates_sha256') == digest(candidates_path):
            print('Already completed; verified candidate cache is reused.')
            return

    payload = {'status': 'running', 'purpose': 'single_frame_candidate_generation_engineering_check', 'is_dataset_ap_result': False, 'protocol': protocol, 'fingerprint': fingerprint}
    atomic_json(output, payload)
    try:
        import numpy as np
        from PIL import Image
        import scipy.io
        import torch
        sys.path[:0] = [str(source / p) for p in ['models', 'utils', 'pointnet2', 'knn']]
        from graspnet import GraspNet, pred_decode
        from data_utils import CameraInfo, create_point_cloud_from_depth_image

        random.seed(protocol['seed'])
        np.random.seed(protocol['seed'])
        torch.manual_seed(protocol['seed'])
        torch.cuda.manual_seed_all(protocol['seed'])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA operators are required.')
        rgb = np.asarray(Image.open(inputs['rgb']).convert('RGB'))
        depth = np.asarray(Image.open(inputs['depth']))
        meta = scipy.io.loadmat(inputs['meta'], variable_names=['intrinsic_matrix', 'factor_depth'])
        intrinsic = np.asarray(meta['intrinsic_matrix'], dtype=np.float64)
        factor_depth = float(np.asarray(meta['factor_depth']).item())
        if depth.ndim != 2 or rgb.shape[:2] != depth.shape:
            raise ValueError('RGB and depth dimensions do not match.')
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all() or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0 or not np.isfinite(factor_depth) or factor_depth <= 0:
            raise ValueError('Invalid depth calibration.')
        camera = CameraInfo(depth.shape[1], depth.shape[0], intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2], factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        valid = np.isfinite(cloud).all(-1) & (cloud[..., 2] > 0) & (cloud[..., 2] <= protocol['maximum_depth_m'])
        points = cloud[valid].astype(np.float32)
        if len(points) < protocol['num_points']:
            raise ValueError('Insufficient valid points; no implicit replacement sampling.')
        sampled_indices = np.random.choice(len(points), protocol['num_points'], replace=False)
        sampled_points = points[sampled_indices]

        net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4, cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        load_result = net.load_state_dict(checkpoint['model_state_dict'], strict=True)
        net = net.cuda().eval()
        tensor = torch.from_numpy(sampled_points).unsqueeze(0).cuda()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            end_points = net({'point_clouds': tensor})
            decoded = pred_decode(end_points)[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        candidates = decoded.detach().cpu().numpy()
        if candidates.ndim != 2 or candidates.shape[1] != 17 or len(candidates) == 0 or not np.isfinite(candidates).all():
            raise ValueError('Empty, non-finite or malformed candidate output.')
        rotations = candidates[:, 4:13].reshape(-1, 3, 3)
        orthogonality_error = np.max(np.abs(np.matmul(rotations.transpose(0, 2, 1), rotations) - np.eye(3)))
        determinants = np.linalg.det(rotations)
        checks = {
            'all_values_finite': bool(np.isfinite(candidates).all()),
            'rotations_orthogonal': bool(orthogonality_error < 1e-4),
            'rotations_right_handed': bool(np.max(np.abs(determinants - 1.0)) < 1e-4),
            'widths_in_0_to_0p1_m': bool(((candidates[:, 1] >= 0) & (candidates[:, 1] <= 0.100001)).all()),
            'heights_0p02_m': bool(np.allclose(candidates[:, 2], 0.02)),
            'depths_in_official_bins': bool(np.isclose(candidates[:, 3, None], [0.01, 0.02, 0.03, 0.04], atol=1e-6).any(1).all()),
            'centers_within_fixed_depth_range': bool(((candidates[:, 15] > 0) & (candidates[:, 15] <= 2.0)).all()),
            'object_ids_unspecified': bool((candidates[:, 16] == -1).all())
        }
        if not all(checks.values()):
            raise ValueError('Candidate geometry check failed: {}'.format(checks))
        temporary = candidates_path.with_suffix('.npy.tmp')
        with temporary.open('wb') as f:
            np.save(f, candidates, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(candidates_path)
        payload.update({
            'status': 'passed', 'completed_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'strict_model_load': str(load_result), 'checkpoint_epoch': int(checkpoint['epoch']),
            'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
            'input_shape_hw': list(depth.shape), 'factor_depth': factor_depth, 'intrinsic_matrix': intrinsic.tolist(),
            'valid_points_before_sampling': int(len(points)), 'sampled_indices_sha256': hashlib.sha256(sampled_indices.tobytes()).hexdigest(),
            'candidate_count': int(len(candidates)), 'candidate_columns': ['score', 'width_m', 'height_m', 'depth_m', 'rotation_row_major_9', 'translation_camera_xyz_m', 'object_id'],
            'candidates_file': str(candidates_path.relative_to(root)), 'candidates_sha256': digest(candidates_path),
            'checks': checks, 'maximum_rotation_orthogonality_error': float(orthogonality_error),
            'rotation_determinant_range': [float(determinants.min()), float(determinants.max())],
            'score_range': [float(candidates[:, 0].min()), float(candidates[:, 0].max())],
            'forward_and_decode_seconds': elapsed, 'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'limitations': ['单帧工程推理，不是数据集AP或机械臂执行成功率。', '未进行碰撞检测和非极大值抑制；输出为原始候选。', '固定随机种子和确定性设置不保证自定义CUDA算子跨设备逐比特相同。', '模型字节已与公开镜像固定哈希核验，未取得官方哈希。']
        })
        atomic_json(output, payload)
        print(json.dumps({k: payload[k] for k in ['status', 'candidate_count', 'forward_and_decode_seconds', 'cuda_peak_allocated_bytes', 'checks']}, ensure_ascii=False))
    except Exception as exc:
        payload.update({'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
        atomic_json(output, payload)
        raise


if __name__ == '__main__':
    main()
