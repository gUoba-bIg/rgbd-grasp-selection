"""Read-only frozen P5/P7 cache adapter for the fixed v51 50-case scope.

No prediction, training, ground-truth labels, AP, cache writes or point-cloud
filtering occur here. Cached receipt metrics are deliberately not returned.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image
from src.p6_local_features_v50 import instance_maps

ROOT = Path(__file__).resolve().parents[1]
AUDIT = 'logs/joint_validation_v51/p5_asset_audit.json'
SCENES = (2, 6, 7, 11, 13, 15, 18, 22, 33, 39)
CONDITIONS = ('clean', 'noise5', 'missing15', 'shift3', 'joint')
FRAME = '0064'
SHAPE = (480, 640)


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def identity(path):
    path = Path(path)
    st = path.stat()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    require(path.stat().st_size == st.st_size and path.stat().st_mtime_ns == st.st_mtime_ns,
            'Source changed during read')
    return dict(sha256=digest, bytes=st.st_size, mtime_ns=st.st_mtime_ns)


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _audit(root):
    audit = _json(root/AUDIT)
    require(audit['passed'] is True, 'Asset audit is not passed')
    keys = [(r['scene'], r['frame'], r['condition']) for r in audit['cases']]
    expected = {(s, FRAME, c) for s in SCENES for c in CONDITIONS}
    require(len(keys) == 50 and len(set(keys)) == 50 and set(keys) == expected,
            'Audit does not bind exact ten-scene, five-condition scope')
    return audit


def _verified(root, relative, bindings, used):
    require(relative in bindings, 'Unbound source: '+relative)
    path = (root/relative).resolve()
    require(path.is_relative_to(root.resolve()), 'Source outside project')
    actual = identity(path)
    require(actual == bindings[relative], 'Frozen source identity differs: '+relative)
    used[relative] = actual
    return path


def _readonly(array):
    value = np.asarray(array)
    value.flags.writeable = False
    return value


def _probability(value, name):
    require(value.shape == SHAPE and np.isfinite(value).all()
            and np.all((value >= 0) & (value <= 1)), 'Invalid '+name)


def _arm(root, record, arm, audit, used):
    spec = record['rgb_anchor'] if arm == 'rgb' else record['p5'][arm]
    path = _verified(root, spec['npz'], audit['files_checked'], used)
    receipt = _json(_verified(root, spec['receipt'], audit['files_checked'], used))
    require(receipt['scene'] == record['scene'] and receipt['frame'] == FRAME,
            'P5 cache frame differs')
    require(receipt['cache_sha256'] == used[spec['npz']]['sha256'], 'P5 NPZ seal differs')
    if arm != 'rgb':
        require(receipt['set']['arm'] == arm and receipt['set']['condition'] == record['condition'],
                'P5 arm/condition differs')
    else:
        require(receipt['arm'] == 'rgb', 'Pure RGB parent differs')
    with np.load(path, allow_pickle=False) as pack:
        names = ('instances', 'foreground_probability', 'center_probability', 'offsets', 'foreground')
        values = {name: pack[name].copy() for name in names}
        if arm != 'rgb':
            values['gate'] = pack['gate'].copy()
    labels = values['instances']
    require(labels.shape == SHAPE and np.issubdtype(labels.dtype, np.integer)
            and np.all(labels >= 0), 'Invalid instance labels')
    for name in ('foreground_probability', 'center_probability'):
        _probability(values[name], name)
    require(values['offsets'].shape == (2, *SHAPE) and np.isfinite(values['offsets']).all(),
            'Invalid offsets')
    require(values['foreground'].shape == SHAPE and values['foreground'].dtype == np.bool_,
            'Invalid foreground mask')
    if 'gate' in values:
        _probability(values['gate'], 'gate')
    scores = [{'instance_id': x['instance_id'], 'score': x['score']} for x in receipt['scores']]
    maps = instance_maps(labels, values['foreground_probability'], scores)
    values.update(maps)
    values['scores'] = scores
    for key, value in tuple(values.items()):
        if isinstance(value, np.ndarray):
            values[key] = _readonly(value)
    return values, receipt.get('corruption')


def dependency_paths(root=ROOT):
    """Return sealed cache/metadata/input dependencies, without decoding arrays."""
    root = Path(root)
    audit = _audit(root)
    # The original audit also binds source, weight, label and raw provenance.
    # Listing a label dependency is not a model-input read.
    return sorted(set(audit['files_checked']) | {AUDIT,
        'logs/joint_validation_v51/p5_asset_audit_full_protected.json',
        'configs/p5_completion_v39.json', 'configs/risk_calibration_v31.json',
        'logs/P5/completion_v39/authorization.json'})


def load_case(scene, condition, root=ROOT):
    """Load one fixed cached case; never select/drop/reorder its candidates.

    candidates and pixels contain all original N rows; pool_indices is the
    immutable twenty-row selection shared by A/B/C/D. rgb is bilinear RGB/255.
    Old P7 instance labels and old risk features are intentionally excluded.
    The shared observed cloud inputs are not masked by either segmentation arm.
    """
    require(type(scene) is int and scene in SCENES and condition in CONDITIONS,
            'Case outside fixed v51 scope')
    root = Path(root).resolve()
    audit = _audit(root)
    record = next(r for r in audit['cases'] if r['scene'] == scene and r['condition'] == condition)
    used = {AUDIT: identity(root/AUDIT)}
    source = record['p7']
    observation_path = _verified(root, source['observations'], audit['files_checked'], used)
    folder = source['folder']
    generation = _json(_verified(root, folder+'/generation.json', audit['files_checked'], used))
    pool_path = _verified(root, source['pool'], audit['files_checked'], used)
    pool = _json(pool_path)
    for path, value in ((folder+'/generation.json', generation), (source['pool'], pool)):
        seal = _json(_verified(root, path+'.receipt.json', audit['files_checked'], used))
        require(seal == used[path], 'P7 JSON receipt differs')
        case = value['case']
        require(case == dict(split='validation', scene=scene, frame=64, condition=condition),
                'P7 case identity differs')
    require(generation['artifact'] == used[source['observations']], 'P7 observation seal differs')
    require(pool['all_methods_same_pool'] is True and pool['GT_used'] is False,
            'Pool is not observation-only/shared')
    indices = np.asarray(pool['indices'], np.int64)
    require(pool['indices'] == source['pool_indices'] and len(indices) == 20
            and len(set(indices.tolist())) == 20, 'Pool identity/cardinality differs')
    with np.load(observation_path, allow_pickle=False) as pack:
        names = ('candidates', 'pixels', 'observed_depth', 'observed_valid', 'K', 'original_size',
                 'sigma', 'quality_valid', 'quality_anomaly')
        observation = {name: pack[name].copy() for name in names}
    candidates, pixels = observation['candidates'], observation['pixels']
    n = source['candidate_count']
    require(candidates.shape == (n, 17) and np.isfinite(candidates).all(), 'Invalid candidates')
    require(pixels.shape == (n,) and np.issubdtype(pixels.dtype, np.integer)
            and np.all((pixels >= 0) & (pixels < np.prod(SHAPE))), 'Invalid seed pixels')
    require(np.all((indices >= 0) & (indices < n)), 'Pool index outside candidate array')
    depth, valid = observation['observed_depth'], observation['observed_valid']
    require(depth.shape == SHAPE and np.isfinite(depth).all() and np.all(depth >= 0), 'Invalid observed depth')
    require(valid.shape == SHAPE and valid.dtype == np.bool_, 'Invalid observed validity')
    k, size = observation['K'], observation['original_size']
    require(k.shape == (3, 3) and np.isfinite(k).all() and k[0, 0] > 0 and k[1, 1] > 0,
            'Invalid intrinsic matrix')
    require(size.shape == (2,) and np.issubdtype(size.dtype, np.integer)
            and np.all(size > 0), 'Invalid original size')
    for name in ('sigma', 'quality_valid', 'quality_anomaly'):
        require(observation[name].shape == SHAPE and np.isfinite(observation[name]).all(),
                'Invalid common quality descriptor')
    arms = {}
    for arm in ('fixed', 'quality', 'rgb'):
        arms[arm], corruption = _arm(root, record, arm, audit, used)
        if arm != 'rgb':
            require(corruption == generation['corruption'], 'P5/P7 corrupted observation differs')
    raw = next(r for r in audit['raw_inputs'] if r['scene'] == scene and r['frame'] == FRAME)
    rgb_relative = raw['raw']['rgb']['p5_path']
    rgb_path = _verified(root, rgb_relative, audit['files_checked'], used)
    require(used[rgb_relative]['sha256'] == raw['raw']['rgb']['sha256'], 'RGB raw identity differs')
    with Image.open(rgb_path) as image:
        require(image.size == tuple(map(int, size)), 'RGB and P7 original grid differ')
        rgb = np.asarray(image.convert('RGB').resize((640, 480), Image.Resampling.BILINEAR),
                         dtype=np.float64)/255.
    return dict(scene=scene, frame=FRAME, condition=condition,
                pool_indices=_readonly(indices), **{k:_readonly(v) for k,v in observation.items()},
                rgb=_readonly(rgb), arms=arms,
                pure_rgb_maps={k:arms['rgb'][k] for k in ('foreground_probability',
                    'instance_confidence', 'soft_boundary', 'boundary_proximity')},
                source_identities=used, corruption=dict(generation['corruption']),
                candidate_order='original_N_rows; apply pool_indices without reordering',
                obstacle_policy='all observed depth points; no segmentation cropping',
                ground_truth_model_inputs=False, new_neural_forwards=0)


def target_association(instances, foreground_probability, seed_pixel, closing_pixels):
    """g=0 association, using every closing pixel in the unchanged denominator.

    A predicted instance ID is local to this mask. No common object-ID meaning
    across methods is assumed. Zero/empty predictions remain failures, not drops.
    """
    labels = np.asarray(instances)
    fg = np.asarray(foreground_probability, np.float64)
    require(labels.ndim == 2 and labels.shape == fg.shape
            and np.issubdtype(labels.dtype, np.integer) and np.all(labels >= 0), 'Invalid association labels')
    require(np.isfinite(fg).all() and np.all((fg >= 0) & (fg <= 1)), 'Invalid association foreground')
    require(isinstance(seed_pixel, (int, np.integer)) and 0 <= int(seed_pixel) < labels.size,
            'Invalid association seed')
    pixels = np.asarray(closing_pixels)
    require(pixels.ndim == 1 and (pixels.size == 0 or np.issubdtype(pixels.dtype, np.integer)),
            'Closing pixel indices must be integers')
    pixels = pixels.astype(np.int64)
    require(np.all((pixels >= 0) & (pixels < labels.size)) and len(np.unique(pixels)) == len(pixels),
            'Invalid/duplicate closing pixel index')
    ident = int(labels.ravel()[int(seed_pixel)])
    support = fg.ravel()[pixels]*(labels.ravel()[pixels] == ident) if ident else np.zeros(len(pixels))
    return dict(target_id=ident, unknown=ident == 0, closing_count=len(pixels),
                empty_closing=len(pixels) == 0,
                association=float(support.mean()) if len(pixels) else 0.,
                candidate_retained=True)
