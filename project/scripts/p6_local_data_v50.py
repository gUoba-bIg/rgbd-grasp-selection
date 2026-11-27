"""Frozen-asset RGB inference and observation-only data adapter for v50.

Importing this module performs no model loading, image inference, or data read.
Formal calls must be owned and budgeted by the v50 controller. Real forward
intent/return events are emitted at the actual model invocation boundary.
"""
import hashlib
import json
import os
import uuid
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.io import loadmat
from src.p6_local_features_v50 import instance_maps, local_features, require

ROOT = Path(__file__).resolve().parents[1]
P4_CONFIG = 'configs/p4_full_baseline_v22r2.json'
P4_SUMMARY = 'results/P4/full_baseline_v22r2/training_and_export.json'
RGB_CHECKPOINT = 'experiments/p4_full_baseline_v22r2/selected/epoch_33_rgb.pt'
RGB_CHECKPOINT_SHA256 = 'd48ccf2a5797bbc4724e412af618f2ad77bd97d8b709329809780b58aeef98f9'
DATA_AUDIT = 'logs/P6/correspondence_v40/data_audit.json'
DECODER = dict(foreground_threshold=.5, center_threshold=.35, min_pixels=20, max_centers=64)
HEAD_NAMES = ('foreground_probability', 'center_probability', 'offsets')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _identity(path):
    path = Path(path); st = path.stat()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(sha256=digest, bytes=st.st_size, mtime_ns=st.st_mtime_ns)


def asset_bindings():
    """Identity/metadata audit only; does not decode an image or run a network."""
    config, summary, data = (_read(ROOT/p) for p in (P4_CONFIG, P4_SUMMARY, DATA_AUDIT))
    require(summary['selected_epoch'] == 33 and config['decoder'] == DECODER
            and config['input_size'] == [480, 640], 'Frozen RGB asset selection differs')
    identity = _identity(ROOT/RGB_CHECKPOINT)
    require(identity['sha256'] == RGB_CHECKPOINT_SHA256, 'RGB checkpoint identity differs')
    covered, missing = {}, {}
    cache_root = ROOT/'results/P4/full_baseline_v22r2/exports/rgb'
    for split, records in data['records'].items():
        covered[split], missing[split] = [], []
        for record in records:
            key = f"scene_{record['scene']:04d}_{record['frame']}"
            cache = cache_root/(key+'.npz')
            if cache.exists():
                receipt = _read(cache.with_suffix('.json'))
                require(_identity(cache)['sha256'] == receipt['cache_sha256'], 'Existing RGB cache changed')
                covered[split].append(dict(key=key, path=cache.relative_to(ROOT).as_posix(),
                                           identity=_identity(cache)))
            else:
                missing[split].append(key)
    paths = [P4_CONFIG, P4_SUMMARY, RGB_CHECKPOINT, DATA_AUDIT,
             'src/segmentation.py', 'scripts/extract_mask_scores_v17.py',
             'scripts/evaluate_segmentation_baseline_v16.py',
             'src/uncertainty_mixture_v43.py']
    return dict(family='frozen_p4_v22r2_rgb_epoch33', checkpoint=RGB_CHECKPOINT,
        checkpoint_identity=identity, input_size=[480, 640], input_mode='RGB_only',
        preprocessing='Pillow RGB bilinear 640x480; float32 /255; ImageNet mean/std',
        decoder=DECODER, dependencies={p:_identity(ROOT/p) for p in paths},
        existing_trusted_cache_coverage=covered, missing_unique_rgb=missing,
        production_unique_rgb_cap=sum(map(len, missing.values())),
        full_replay_rgb_cap=sum(len(x) for x in data['records'].values()),
        segmentation_training_scenes=sorted({r['scene'] for r in config['train_records']}),
        whole_pipeline_scene_independence=False,
        exposure_note='All twenty current scenes were exposed to upstream fixed P4 segmentation training; only new local-head train/aux roles are separated.',
        new_real_rgb_forward_calls=0)


def rgb_inputs(rgb):
    color = np.asarray(rgb)
    require(color.ndim == 3 and color.shape[-1] == 3 and np.isfinite(color).all()
            and np.all((color >= 0) & (color <= 1)), 'RGB must be finite [0,1]')
    color = color.astype(np.float32)
    color = (color-np.asarray([.485, .456, .406], np.float32))/np.asarray([.229, .224, .225], np.float32)
    return np.ascontiguousarray(color.transpose(2, 0, 1))


def load_rgb_model(device='cpu'):
    require(device == 'cpu', 'v50 frozen RGB inference device is CPU')
    require(_identity(ROOT/RGB_CHECKPOINT)['sha256'] == RGB_CHECKPOINT_SHA256, 'RGB checkpoint changed')
    import torch
    from src.segmentation import PilotInstanceNet
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = PilotInstanceNet(mode='rgb').to(device)
    payload = torch.load(ROOT/RGB_CHECKPOINT, map_location='cpu', weights_only=False)
    require(payload['epoch'] == 33 and payload['arm'] == 'rgb', 'Wrong selected RGB model')
    model.load_state_dict(payload['model'], strict=True)
    model.eval()
    return model


def decode_heads(heads):
    # The frozen decoder imports an old module with a four-thread environment.
    # Restore the v50 two-thread process contract immediately; no model is run.
    from scripts.extract_mask_scores_v17 import decode_with_scores
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    labels, foreground, scores, _ = decode_with_scores(heads, **DECODER)
    maps = instance_maps(labels, heads['foreground_probability'], scores)
    return dict(instances=labels, foreground=foreground, **heads,
                instance_confidence=maps['instance_confidence'], soft_boundary=maps['soft_boundary'],
                boundary_proximity=maps['boundary_proximity'])


def infer_rgb(model, rgb, key, on_event=lambda record: None):
    """One real CPU network invocation; no retry, no labels/depth are accepted.

    A model exception leaves its intent without a return. A decoder failure
    after a successful forward preserves the already returned network event.
    The caller persists events and the full output arrays atomically.
    """
    import torch
    require(np.asarray(rgb).shape == (480, 640, 3), 'Frozen RGB inference grid differs')
    require(not model.training and next(model.parameters()).device.type == 'cpu', 'RGB model must be CPU/eval')
    require(torch.get_num_threads() == 2, 'RGB inference CPU thread count differs')
    inputs = torch.from_numpy(rgb_inputs(rgb))[None]
    call_id = uuid.uuid4().hex
    on_event(dict(stage='rgb_forward', event='intent', id=call_id, key=str(key),
                  input_shape=list(inputs.shape), device='cpu', batch_size=1))
    with torch.inference_mode():
        output = model(inputs)
    on_event(dict(stage='rgb_forward', event='returned', id=call_id, key=str(key),
                  device='cpu', batch_size=1))
    require(set(output) == {'foreground', 'centers', 'offsets'}
            and all(x.dtype == torch.float32 and bool(torch.isfinite(x).all()) for x in output.values()),
            'RGB model returned invalid FP32 heads')
    heads = dict(foreground_probability=output['foreground'][0, 0].sigmoid().cpu().numpy(),
                 center_probability=output['centers'][0, 0].sigmoid().cpu().numpy(),
                 offsets=output['offsets'][0].cpu().numpy())
    return decode_heads(heads)


def load_observed_frame(data, record):
    """Read RGB/depth/camera only, matching v40's exact input preprocessing."""
    directory = ROOT/data['data_root']
    meta = loadmat(directory/record['files']['meta'], variable_names=['intrinsic_matrix', 'factor_depth'])
    intrinsic = np.asarray(meta['intrinsic_matrix'], np.float64)
    factor = float(meta['factor_depth'].reshape(-1)[0])
    require(intrinsic.shape == (3, 3) and np.isfinite(intrinsic).all()
            and np.isfinite(factor) and factor > 0, 'Invalid observed camera')
    with Image.open(directory/record['files']['depth']) as im:
        original_size = im.size
        raw = np.asarray(im.resize((640, 480), Image.Resampling.NEAREST), np.float32)/factor
    valid = np.isfinite(raw) & (raw > 0)
    depth = np.where(valid, np.minimum(raw, 2.), 0.).astype(np.float32)
    with Image.open(directory/record['files']['rgb']) as im:
        require(im.size == original_size, 'RGB/depth source grids differ')
        rgb = np.asarray(im.convert('RGB').resize((640, 480), Image.Resampling.BILINEAR), np.float64)/255.
    return dict(rgb=rgb, depth=depth, valid=valid, eligible=valid & (raw <= 2.),
                K=intrinsic, original_size=original_size)


def frame_features(frame, observed_depth, observed_valid, predicted_maps):
    return local_features(frame['rgb'], observed_depth, observed_valid, predicted_maps,
                          frame['K'], frame['original_size'])
