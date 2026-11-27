"""Verify the single same-client rigid-body damping write and its complete API trail."""
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / 'IP140_R1_smoke10_slot07_nominal021_01/static_precheck'
BINARY = ROOT / 'runtime/simulation_v52e/venv/Lib/site-packages/pybullet.cp312-win_amd64.pyd'
BINARY_SHA256 = '882cd523a255e66c94d5b91c119f9c688fdbc0956161eff30f2171e21677ab49'
OBJECT_URDF = 'data/simulation_v52/objects/final_static_audit/021/object.urdf'
BASE_LINK = -1
DIRECT_MODE = 2
DAMPING = ('linearDamping', 'angularDamping')
TARGETS = {'lateralFriction': 0.6, 'restitution': 0.0, 'linearDamping': 0.04, 'angularDamping': 0.04}


def _digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_proof():
    evidence = SOURCE / 'damping_readback_blocker.json'
    excerpt = SOURCE / 'damping_sources.txt'
    historical = json.loads(evidence.read_bytes())
    server = next(x for x in historical['source_evidence'] if x['id'] == 'server_damping_setters')
    src = ROOT / 'runtime/simulation_v52e/src/pybullet-3.2.7/examples'
    files = (
        (evidence, '004155e652c4099d15fe5889c67da515cc25d02bc53be1ccac254109170fd4bb', None),
        (excerpt, '18c14d1cf84e11252d1ebc24da8d60d2f2b20c55649564ddd222dbe9d012b12f', None),
        (ROOT / server['path'], server['sha256'], [server['start_line'], server['end_line']]),
        (src / 'pybullet/pybullet.c', '9a411aa30d3f95fa15bce4099ad5cf0ee1dae58c987f7bb02c029637ab78698c', [1519, 1526]),
        (src / 'SharedMemory/PhysicsClientC_API.cpp', '59e429568ad2f27b4b0a822837147f829822ce9298115ad1dcf77d82d59c2d27', [3469, 3490]),
    )
    proof = []
    for path, expected, lines in files:
        actual = _digest(path)
        if actual != expected:
            raise ValueError(f'damping write provenance mismatch: {path.name}')
        proof.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=actual, lines=lines))
    if historical['binary_sha256'] != BINARY_SHA256 or _digest(BINARY) != BINARY_SHA256:
        raise ValueError('frozen engine binary digest mismatch')
    return proof


def _ids(body, client):
    return all(type(i) is int and i >= 0 for i in (body, client))


def _fields():
    return {
        name: dict(verification_mode='audited_write_path', target=0.04, observed=None,
                   public_readback_available=False, setter_call_count=1, setter_returned_normally=True)
        for name in DAMPING
    }


def _receipt(body, client, sequence, result, sources):
    return dict(object_id='021', object_body=body, link=BASE_LINK, client_id=client,
                api_sequence=sequence, setter_method='changeDynamics', args=[body, BASE_LINK],
                kwargs={**TARGETS, 'physicsClientId': client}, synchronous_return=result,
                verification_mode='audited_write_path', setter_call_count=1, setter_returned_normally=True,
                binary_path=BINARY.relative_to(ROOT).as_posix(), binary_sha256=BINARY_SHA256,
                source_sha256_evidence=sources, fields=_fields())


def apply_damping_write(p, object_body, client_id):
    """p is the session's ApiTrace. The caller archives the raw call even on exception."""
    if not _ids(object_body, client_id):
        raise ValueError('invalid actual object/client identity')
    if Path(p.module.__file__).resolve() != BINARY.resolve():
        raise ValueError('imported PyBullet module is not the frozen binary')
    proof = source_proof()
    sequence = p.sequence + 1
    result = p.changeDynamics(object_body, BASE_LINK, **TARGETS, physicsClientId=client_id)
    if result is not None or p.sequence != sequence:
        raise ValueError('unexpected setter return or sequence')
    return _receipt(object_body, client_id, sequence, result, proof)


def validate_damping_write(receipt, object_body, client_id, api_records):
    """Stream native ApiTrace rows or its lossless relevant-call projection, through the audit point."""
    if not isinstance(receipt, dict) or not _ids(object_body, client_id):
        return ['missing_receipt_or_invalid_identity']
    errors = []
    sequence = receipt.get('api_sequence')
    if type(sequence) is not int or sequence <= 0:
        return ['invalid_setter_sequence']
    expected = _receipt(object_body, client_id, sequence, None, source_proof())
    if receipt != expected:
        errors.append('receipt_identity_parameters_observation_or_provenance')
    if any(type(receipt.get('fields', {}).get(k, {}).get('public_readback_available')) is not bool
           for k in DAMPING):
        errors.append('invalid_readback_flags')
    pending = {}
    connects = []
    loads = []
    setters = []
    first_step = None
    disconnect = None
    target_path = (ROOT / OBJECT_URDF).resolve()
    for row in api_records:
        seq, action = row[:2]
        if action == 'call':
            _, _, _, method, args, kwargs = row
            if method == 'stepSimulation':
                if first_step is None:
                    first_step = seq
                continue
            if method == 'disconnect':
                disconnect = seq
                continue
            if method == 'connect':
                is_direct = (args[0] if args else kwargs.get('connectionMode')) == DIRECT_MODE
                item = dict(seq=seq, direct=is_direct)
                connects.append(item)
            elif method == 'loadURDF':
                path = args[0] if args else kwargs.get('fileName')
                if not isinstance(path, str) or (ROOT / path).resolve() != target_path:
                    continue
                item = dict(seq=seq, client=kwargs.get('physicsClientId'))
                loads.append(item)
            elif method == 'changeDynamics':
                body = args[0] if args else kwargs.get('bodyUniqueId')
                if body != object_body:
                    continue
                item = dict(seq=seq, args=list(args), kwargs=kwargs)
                setters.append(item)
            else:
                continue
            pending[seq] = item
        elif action in ('return', 'error') and seq in pending:
            item = pending.pop(seq)
            item.update(returned=(action == 'return'), value=row[3])
    if len(connects) != 1 or not all(
        c['direct'] and c.get('returned') and type(c['value']) is int
        and c['value'] == client_id and c['seq'] < sequence for c in connects
    ):
        errors.append('DIRECT_connection_identity_or_order')
    if len(loads) != 1 or not all(
        a.get('returned') and a['client'] == client_id and type(a['value']) is int
        and a['value'] == object_body and a['seq'] < sequence for a in loads
    ):
        errors.append('object_load_identity_or_order')
    if len(setters) != 1:
        errors.append('object_changeDynamics_count')
    for write in setters:
        if write['seq'] != sequence or write['args'] != expected['args'] or write['kwargs'] != expected['kwargs']:
            errors.append('setter_arguments_or_sequence')
        if not write.get('returned') or write.get('value') is not None:
            errors.append('setter_exception_missing_or_abnormal_return')
    if (first_step is not None and first_step <= sequence) or (disconnect is not None and disconnect <= sequence):
        errors.append('setter_outside_zero_step_connection')
    return sorted(set(errors))

