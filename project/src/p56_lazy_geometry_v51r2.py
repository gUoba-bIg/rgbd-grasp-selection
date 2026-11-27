"""Resource-only adapter for the frozen official GraspNet evaluator.

No physical equation, candidate, mesh coordinate, SDF or scientific threshold is
changed. OBJ storage is preallocated, while the original constructors and score
function are retained. Model loads and physical score calls are separate events.
"""
from __future__ import annotations

import gc
import hashlib
import inspect
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.p8_geometry_v32 import Geometry
from scripts.p6_common_v24 import ROOT, read, stat

CHECK_EVERY = 16384
CACHE_BYTES_CAP = 2 * 1024 ** 3
OBJECT46_CACHE = ROOT/'results/joint_validation_v51r2/assets/object_046'


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _rss():
    import psutil
    memory = psutil.Process().memory_info()
    return dict(rss_bytes=int(memory.rss),
                process_peak_rss_bytes=int(getattr(memory, 'peak_wset', memory.rss)))


def _tick(guard):
    if guard is not None:
        guard()


def _metadata(path):
    info = Path(path).stat()
    return info.st_size, info.st_mtime_ns


def _array_identity(value):
    if value is None:
        return None
    digest = hashlib.sha256()
    # Bounded views; do not create another full-size byte string.
    for start in range(0, len(value), CHECK_EVERY):
        digest.update(np.ascontiguousarray(value[start:start + CHECK_EVERY]).view(np.uint8))
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=int(value.nbytes),
                sha256=digest.hexdigest())


def read_obj_arrays(path, *, guard=None):
    """Two-pass, original-token-semantics OBJ reader, with no mesh construction.

    Uniform-width face rows are retained (including quads); no triangulation,
    normal repair, coordinate scaling or interpretation of negative indices.
    Ragged rows fail, as the pinned NumPy/Mesh3D path also fails on these rows.
    """
    path = Path(path)
    if path.suffix != '.obj':
        raise ValueError('Extension %s invalid for OBJs' % path.suffix)
    before = _metadata(path)
    counts = dict(v=0, vn=0, f=0)
    widths = dict(v=None, vn=None, f=None)
    minimum_index = maximum_index = None
    with path.open('r') as source:
        for line_number, line in enumerate(source, 1):
            if line_number % CHECK_EVERY == 0:
                _tick(guard)
            fields = line.split()
            if fields and fields[0] in counts:
                tag = fields[0]
                width = len(fields[1:4]) if tag != 'f' else len(fields) - 1
                if widths[tag] is None:
                    widths[tag] = width
                _require(widths[tag] == width, 'Ragged OBJ rows cannot be converted by pinned Mesh3D')
                counts[tag] += 1
                if tag == 'f':
                    slash = '/' in fields[1]
                    for token in fields[1:]:
                        index = int(token.split('/')[0] if slash else token) - 1
                        minimum_index = index if minimum_index is None else min(minimum_index, index)
                        maximum_index = index if maximum_index is None else max(maximum_index, index)
    _require(_metadata(path) == before, 'OBJ changed during count pass')
    _tick(guard)
    arrays = {}
    for tag in ('v', 'vn', 'f'):
        if tag == 'vn' and not counts[tag]:
            arrays[tag] = None
        else:
            shape = (counts[tag], widths[tag]) if counts[tag] else (0,)
            # np.array(Python-int rows) uses int32 on pinned Windows NumPy,
            # promoting to int64 for larger integers. Match that inference.
            face_dtype = np.asarray([minimum_index, maximum_index]).dtype if minimum_index is not None else np.dtype(float)
            arrays[tag] = np.empty(shape, dtype=face_dtype if tag == 'f' and counts[tag] else np.float64)
    at = dict(v=0, vn=0, f=0)
    with path.open('r') as source:
        for line_number, line in enumerate(source, 1):
            if line_number % CHECK_EVERY == 0:
                _tick(guard)
            fields = line.split()
            if not fields:
                continue
            tag = fields[0]
            if tag in ('v', 'vn'):
                row = [float(value) for value in fields[1:4]]
            elif tag == 'f':
                # The original branch is chosen by the FIRST face token only.
                if '/' not in fields[1]:
                    row = [int(value) - 1 for value in fields[1:]]
                else:
                    row = []
                    for value in fields[1:]:
                        parts = value.split('/')
                        row.append(int(parts[0]) - 1)
                        if len(parts) > 1 and parts[1] != '':
                            int(parts[1])
                        if len(parts) > 2:
                            int(parts[2])
                        # The pinned parser ignores fourth and later fields.
            else:
                continue
            arrays[tag][at[tag]] = row
            at[tag] += 1
    _require(at == counts and _metadata(path) == before, 'OBJ changed during fill pass')
    _tick(guard)
    return arrays['v'], arrays['f'], arrays['vn'], dict(
        counts=counts, widths=widths, arrays={k: _array_identity(v) for k, v in arrays.items()},
        raw_array_bytes=sum(a.nbytes for a in arrays.values() if a is not None),
        parser='two_pass_original_float_int_semantics_no_geometry_change')


def verify_obj_arrays_independently(path, vertices, triangles, normals, *, guard=None):
    """Third streaming pass: independent original-loop conversion, every value.

    This deliberately does not call read_obj_arrays or its parsing expressions.
    It follows the pinned ObjFile.read loop, discarding each verified row. No
    second full mesh/list or sampled/hash-only comparison is substituted.
    """
    before = _metadata(path)
    counts = dict(v=0, vn=0, f=0)
    values = dict(v=vertices, vn=normals, f=triangles)
    _require(vertices.dtype == np.dtype(float) and
             (normals is None or normals.dtype == np.dtype(float)),
             'Independent OBJ original float dtype mismatch')
    lowest = highest = None
    with Path(path).open('r') as stream:
        for line_number, line in enumerate(stream, 1):
            if line_number % CHECK_EVERY == 0:
                _tick(guard)
            vals = line.split()
            if not vals:
                continue
            if vals[0] == 'v' or vals[0] == 'vn':
                parsed = list(map(float, vals[1:4]))
            elif vals[0] == 'f':
                vi, vti, nti = [], [], []
                if vals[1].find('/') == -1:
                    vi = list(map(int, vals[1:]))
                    vi = [i - 1 for i in vi]
                else:
                    for j in range(1, len(vals)):
                        tokens = vals[j].split('/')
                        for i in range(len(tokens)):
                            if i == 0:
                                vi.append(int(tokens[i]) - 1)
                            elif i == 1:
                                if tokens[i] != '':
                                    vti.append(int(tokens[i]))
                            elif i == 2:
                                nti.append(int(tokens[i]))
                parsed = vi
                for integer in vi:
                    lowest = integer if lowest is None else min(lowest, integer)
                    highest = integer if highest is None else max(highest, integer)
            else:
                continue
            tag = vals[0]
            array = values[tag]
            _require(array is not None and counts[tag] < len(array), 'Independent OBJ row count mismatch')
            # Compare floating-point bit patterns (including signed zero), not a tolerance.
            expected = np.asarray(parsed, dtype=array.dtype)
            actual = array[counts[tag]]
            _require(expected.shape == actual.shape and expected.tobytes() == actual.tobytes(),
                     'Independent OBJ value mismatch at line ' + str(line_number))
            counts[tag] += 1
    _require(counts == {k: 0 if a is None else len(a) for k, a in values.items()},
             'Independent OBJ total count mismatch')
    _require(normals is not None or counts['vn'] == 0, 'Absent normals changed')
    original_face_dtype = np.array([lowest, highest]).dtype if lowest is not None else np.dtype(float)
    _require(triangles.dtype == original_face_dtype, 'Independent OBJ original face dtype mismatch')
    _require(_metadata(path) == before, 'OBJ changed during independent verification')
    _tick(guard)
    return dict(passed=True, all_rows_and_all_values=True, counts=counts,
                independent_conversion=True, comparison='exact_dtype_bytes_per_row',
                full_duplicate_mesh_constructed=False)


def _same_array(left, right, name):
    _require((left is None) == (right is None), 'Constructor changed missing ' + name)
    if left is None:
        return
    _require(left.shape == right.shape and left.dtype == right.dtype, 'Constructor changed schema ' + name)
    for first in range(0, len(left), CHECK_EVERY):
        _require(left[first:first + CHECK_EVERY].tobytes() == right[first:first + CHECK_EVERY].tobytes(),
                 'Constructor changed values ' + name)


def write_obj_cache(directory, vertices, triangles, normals, *, source_bindings,
                    parser_proof, verification, guard=None):
    """Persist only exact independently verified raw arrays, never a new mesh."""
    directory = Path(directory).resolve()
    _require(directory.is_relative_to(ROOT.resolve()), 'OBJ cache must remain inside this project')
    _require(not directory.exists(), 'Existing/partial OBJ cache must be preserved, not overwritten')
    _require(verification['passed'] and verification['all_rows_and_all_values']
             and verification['independent_conversion'], 'Cache needs complete independent verification')
    arrays = dict(vertices=vertices, triangles=triangles, normals=normals)
    expected_bytes = sum(a.nbytes + 256 for a in arrays.values() if a is not None)
    _require(expected_bytes + 65536 <= CACHE_BYTES_CAP, 'Lossless raw cache exceeds fixed 2GiB output cap')
    directory.mkdir(parents=True, exist_ok=False)
    entries = {}
    for name, array in arrays.items():
        _tick(guard)
        if array is None:
            entries[name] = None
            continue
        destination = directory/(name + '.npy')
        with destination.open('xb') as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        entries[name] = dict(file=destination.name, identity=stat(destination),
                             shape=list(array.shape), dtype=str(array.dtype))
    receipt = dict(format='v51r2_independently_verified_lossless_obj_arrays',
                   source_bindings=source_bindings, arrays=entries,
                   parser_proof=parser_proof, independent_verification=verification)
    destination = directory/'cache.json'
    with destination.open('x', encoding='utf8') as stream:
        json.dump(receipt, stream, allow_nan=False, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    with (directory/'cache.json.receipt.json').open('x', encoding='utf8') as stream:
        json.dump(stat(destination), stream, allow_nan=False, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    total = sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
    _require(total <= CACHE_BYTES_CAP, 'OBJ cache exceeds fixed output cap')
    return dict(path=directory.relative_to(ROOT).as_posix(), receipt_identity=stat(destination),
                entries=entries, bytes=total)


def read_obj_cache(directory, *, source_bindings, guard=None):
    directory = Path(directory)
    manifest = directory/'cache.json'
    _require(stat(manifest) == read(directory/'cache.json.receipt.json'), 'OBJ cache manifest changed')
    receipt = read(manifest)
    _require(receipt['source_bindings'] == source_bindings, 'OBJ cache frozen source mismatch')
    verification = receipt['independent_verification']
    _require(verification['passed'] and verification['all_rows_and_all_values']
             and verification['independent_conversion'], 'OBJ cache independent proof missing')
    arrays = {}
    _require(set(receipt['arrays']) == {'vertices', 'triangles', 'normals'}, 'OBJ cache array schema changed')
    for name, entry in receipt['arrays'].items():
        _tick(guard)
        if entry is None:
            _require(name == 'normals', 'Only absent OBJ normals can be None')
            arrays[name] = None
            continue
        _require(entry['file'] == name + '.npy', 'OBJ cache path changed')
        path = directory/entry['file']
        _require(stat(path) == entry['identity'], 'OBJ cache array changed')
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        _require(list(array.shape) == entry['shape'] and str(array.dtype) == entry['dtype'],
                 'OBJ cache dtype/shape changed')
        _require(not array.flags.writeable, 'OBJ cache must be read only')
        arrays[name] = array
    return arrays['vertices'], arrays['triangles'], arrays['normals'], receipt


def load_dexnet_preallocated(data_path, *, official, guard=None, on_event=None,
                             independent_verify=False, cache_directory=None,
                             create_cache=False, source_bindings=None):
    """Return (original complete GraspableObject3D, load proof)."""
    from graspnetAPI.utils.dexnet.grasping.meshpy.mesh import Mesh3D
    obj_path, sdf_path = Path(str(data_path) + '.obj'), Path(str(data_path) + '.sdf')
    started = time.monotonic()
    stages = []

    def phase(name, function):
        _tick(guard)
        phase_id = uuid.uuid4().hex
        begin = time.monotonic()
        if on_event:
            on_event(dict(event='intent', kind='asset_phase', id=phase_id, phase=name, **_rss()))
        result = function()
        _tick(guard)
        receipt = dict(event='returned', kind='asset_phase', id=phase_id, phase=name,
                       seconds=time.monotonic() - begin, **_rss())
        stages.append(receipt)
        if on_event:
            on_event(receipt)
        return result

    cache_proof = None
    if cache_directory is not None and not create_cache:
        vertices, triangles, normals, cached = phase('verified_lossless_obj_cache_read',
            lambda: read_obj_cache(cache_directory, source_bindings=source_bindings, guard=guard))
        parser_proof = cached['parser_proof']
        independent = cached['independent_verification']
        cache_proof = dict(path=Path(cache_directory).relative_to(ROOT).as_posix(),
                           receipt_identity=stat(Path(cache_directory)/'cache.json'), reused=True)
    else:
        vertices, triangles, normals, parser_proof = phase('obj_two_pass', lambda: read_obj_arrays(obj_path, guard=guard))
        independent = phase('independent_all_value_verification',
            lambda: verify_obj_arrays_independently(obj_path, vertices, triangles, normals, guard=guard)) if independent_verify else None
        if create_cache:
            _require(cache_directory is not None and source_bindings is not None and independent is not None,
                     'Cache creation requires original source bindings and independent verification')
            cache_proof = phase('verified_lossless_obj_cache_write', lambda: write_obj_cache(
                cache_directory, vertices, triangles, normals, source_bindings=source_bindings,
                parser_proof=parser_proof, verification=independent, guard=guard))
    # The original constructor intentionally retains its default Trimesh,
    # watertight/COM computation, dtype copies, and three-normal-row transpose.
    mesh = phase('original_mesh_constructor', lambda: Mesh3D(vertices, triangles, normals))
    _same_array(mesh.vertices, vertices, 'vertices')
    _same_array(mesh.triangles, triangles, 'triangles')
    expected_normals = normals.T if normals is not None and len(normals) == 3 else normals
    _same_array(mesh.normals, expected_normals, 'normals')
    bbox = (np.min(vertices, axis=0) + np.max(vertices, axis=0)) / 2
    centroid = np.mean(vertices, axis=0)
    _require(np.array_equal(mesh.bb_center, bbox) and np.array_equal(mesh.centroid, centroid),
             'Original constructor derived bbox/centroid differs')
    derived = dict(bb_center=mesh.bb_center.tolist(), centroid=mesh.centroid.tolist(),
                   center_of_mass=mesh.center_of_mass.tolist(),
                   is_watertight=bool(mesh.is_watertight),
                   constructor_arrays_exact=True,
                   original_constructor_defaults_retained=True)
    for name in ('bb_center', 'centroid', 'center_of_mass'):
        _require(np.isfinite(derived[name]).all(), 'Nonfinite derived mesh ' + name)
    del vertices, triangles, normals, expected_normals
    gc.collect()
    sdf = phase('original_sdf_reader', lambda: official.SdfFile(str(sdf_path)).read())
    model = phase('original_graspable_constructor', lambda: official.GraspableObject3D(sdf, mesh))
    proof = dict(passed=True, parser=parser_proof, independent_verification=independent,
                 derived=derived, lossless_cache=cache_proof, stages=stages, seconds=time.monotonic() - started,
                 **_rss())
    return model, proof


@dataclass(frozen=True)
class LazyDexReference:
    owner: object
    object_id: int


class LazyGeometry(Geometry):
    """Frozen geometry + one complete DexNet model loaded only at a real score."""
    def __init__(self, config, *, guard=None, on_event=None, loader=None):
        super().__init__(config)
        self._initialize_lazy(guard=guard, on_event=on_event, loader=loader)

    def _initialize_lazy(self, *, guard=None, on_event=None, loader=None):
        self._lazy_guard = guard
        self._lazy_event = on_event
        self._lazy_loader = loader
        self._lazy_cached_id = None
        self._lazy_cached_model = None
        self._lazy_resolving = False

    def dex(self, oid):
        return LazyDexReference(self, int(oid))

    def release(self):
        prior = self._lazy_cached_id
        self._lazy_cached_model = None
        self._lazy_cached_id = None
        gc.collect()
        if prior is not None and self._lazy_event:
            self._lazy_event(dict(event='released', kind='dexnet_cache', object_id=prior, **_rss()))

    def _load_model(self, oid):
        paths = [self.root/f'models/{oid:03d}/textured.{extension}' for extension in ('obj', 'sdf')]
        for path in paths:
            _require(stat(path) == self.refs[path.relative_to(ROOT).as_posix()], 'Frozen DexNet source changed')
        model, proof = load_dexnet_preallocated(self.root/f'models/{oid:03d}/textured',
            official=self.official, guard=self._lazy_guard, on_event=self._lazy_event,
            cache_directory=OBJECT46_CACHE if oid == 46 else None,
            source_bindings={p.relative_to(ROOT).as_posix():self.refs[p.relative_to(ROOT).as_posix()] for p in paths})
        for path in paths:
            _require(stat(path) == self.refs[path.relative_to(ROOT).as_posix()], 'DexNet source changed while loading')
        return model, proof

    def resolve(self, value):
        _require(isinstance(value, LazyDexReference) and value.owner is self, 'Foreign/nonlazy DexNet reference')
        if self._lazy_cached_id == value.object_id:
            return self._lazy_cached_model
        _require(not self._lazy_resolving, 'Reentrant DexNet loader')
        self.release()
        _tick(self._lazy_guard)
        call_id = uuid.uuid4().hex
        common = dict(id=call_id, kind='dexnet_model_load', object_id=value.object_id)
        if self._lazy_event:
            self._lazy_event(dict(event='intent', **common, **_rss()))
        self._lazy_resolving = True
        start = time.monotonic()
        try:
            model, proof = (self._lazy_loader or self._load_model)(value.object_id)
            _tick(self._lazy_guard)
            self._lazy_cached_model = model
            self._lazy_cached_id = value.object_id
            if self._lazy_event:
                self._lazy_event(dict(event='returned', **common, seconds=time.monotonic() - start,
                                      proof=proof, **_rss()))
            return model
        finally:
            self._lazy_resolving = False

    def oracle(self, normalized, state):
        previous = self.official.get_grasp_score
        def resolved(grasp, model, *args, **kwargs):
            # The tracking wrapper starts the actual physical score only after
            # successful loading. A failed load is not a returned physical call.
            actual = self.resolve(model)
            return previous(grasp, actual, *args, **kwargs)
        self.official.get_grasp_score = resolved
        try:
            return super().oracle(normalized, state)
        finally:
            self.official.get_grasp_score = previous
            self.release()


def asset_preflight(root, guard=None, on_event=None):
    """One bounded asset-only load of object46; caller supplies all resource guards.

    No official eval_grasp/get_grasp_score, NN call, or sample selection occurs.
    Returns only after full parser/constructor verification and memory release.
    Failure preserves emitted intent/partial phase events; it is never converted
    to a successful return. Caller persists the returned report once.
    """
    root = Path(root).resolve()
    _require(root == ROOT.resolve(), 'Preflight must use the designated project root')
    config_path = root/'configs/p8_ablation_v32.json'
    config = read(config_path)
    from graspnetAPI.utils import eval_utils as official
    from graspnetAPI.utils.dexnet.grasping.meshpy import mesh, obj_file, sdf_file
    data_path = root/config['data_root']/'models/046/textured'
    bindings = {}
    for extension in ('obj', 'sdf'):
        path = Path(str(data_path) + '.' + extension)
        binding = stat(path)
        _require(binding == config['reference_inputs'][path.relative_to(root).as_posix()],
                 'Preflight source is not original frozen object46')
        bindings[path.relative_to(root).as_posix()] = binding
    call_id = uuid.uuid4().hex
    start = time.monotonic()
    if on_event:
        on_event(dict(event='intent', id=call_id, kind='asset_preflight', object_id=46, **_rss()))
    model = None
    try:
        _tick(guard)
        model, proof = load_dexnet_preallocated(data_path, official=official, guard=guard,
            on_event=on_event, independent_verify=True, cache_directory=OBJECT46_CACHE,
            create_cache=True, source_bindings=bindings)
        for relative, binding in bindings.items():
            _require(stat(root/relative) == binding, 'Asset changed during preflight')
        _tick(guard)
        report = dict(passed=True, object_id=46, source_bindings=bindings,
            reference_config_identity=stat(config_path), adapter_identity=stat(Path(__file__)),
            adapter_source=stat(Path(__file__)),
            pinned_sources={Path(module.__file__).relative_to(root).as_posix():stat(module.__file__)
                            for module in (mesh, obj_file, sdf_file, official)},
            load_proof=proof, eval_grasp_calls=0, get_grasp_score_calls=0, neural_calls=0,
            seconds=time.monotonic() - start, **_rss())
    finally:
        model = None
        gc.collect()
    _tick(guard)
    report['released'] = True
    report['after_release'] = _rss()
    report['seconds'] = time.monotonic() - start
    if on_event:
        on_event(dict(event='returned', id=call_id, kind='asset_preflight', object_id=46,
                      seconds=report['seconds'], **_rss()))
    return report
