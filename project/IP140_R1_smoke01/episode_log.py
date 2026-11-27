"""Append-only telemetry and budget accounting. No simulator connection on import."""
import gzip
import hashlib
import json
import math
import os
import struct
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def exclusive_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def digest(path):
    with Path(path).open('rb') as fp:
        return hashlib.file_digest(fp,'sha256').hexdigest()


def charge_slot(number, zero_path, registration_path):
    """An intent is a charge even if the worker crashes. No retry entry point."""
    if not 1 <= number <= 12:
        raise ValueError("slot outside approved budget")
    freeze = ROOT/'preregistration.json'
    if Path(registration_path).resolve() != freeze or Path(zero_path).resolve() != ROOT/'zero_readback.json':
        raise RuntimeError("admission paths do not belong to this frozen version")
    if digest(freeze) != "4f62a02c16ecc56c962c108ce58382f801502e91485e475f8f9f032f9878a2c8":
        raise RuntimeError("registration identity mismatch")
    plan=json.loads(freeze.read_text('utf-8'))
    if [s['slot'] for s in plan['slots']]!=list(range(1,13)) or plan['max_starts']!=12:
        raise RuntimeError("invalid twelve-slot manifest")
    if (ROOT/'STOP.json').exists():
        raise RuntimeError("batch sealed")
    gate = json.loads(Path(zero_path).read_text(encoding='utf-8-sig'))
    if gate.get("passed") is not True or gate.get("unverified") != [] or gate.get("mismatches") != []:
        raise RuntimeError("zero readback gate closed")
    if gate.get('model_sha256') != digest(ROOT/'model/ur5e_ip140_r1.urdf') or gate.get('manifest_sha256') != digest(ROOT/'model/manifest.json'):
        raise RuntimeError("unbound model or manifest")
    if gate.get('preregistration_sha256') != digest(registration_path):
        raise RuntimeError("gate not bound to frozen slot configuration")
    base = ROOT/'episodes'
    base.mkdir(exist_ok=True)
    for earlier in range(1,number):
        previous = base/f'{earlier:02d}'/'result.json'
        if not previous.is_file() or json.loads(previous.read_text('utf-8')).get('slot_passed') is not True:
            raise RuntimeError("previous slot failed, incomplete or unknown")
    dest = base/f'{number:02d}'
    dest.mkdir(exist_ok=False)
    exclusive_json(dest/'precharge.json',dict(slot=number,at=timestamp(),pid=os.getpid(),
        registration_sha256=digest(registration_path),zero_report_sha256=digest(zero_path)))
    return dest


def safe_values(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_float":repr(value)}
    if isinstance(value, dict):
        return {k:safe_values(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [safe_values(v) for v in value]
    return value


class Journal:
    def __init__(self, directory):
        self.dir=Path(directory)
        self.stream=gzip.open(self.dir/'telemetry.jsonl.gz','xt',encoding='utf-8')
        self.steps_started=0
        self.steps_returned=0
        self.first_wall=time.monotonic()

    def record(self, event, **fields):
        fields.update(event=event,utc=timestamp(),wall_s=time.monotonic()-self.first_wall)
        self.stream.write(json.dumps(safe_values(fields),ensure_ascii=False,allow_nan=False)+'\n')
        self.stream.flush()

    def intent(self, phase, command):
        if self.steps_started!=self.steps_returned:
            raise RuntimeError("previous step has no return")
        self.steps_started+=1
        self.record("step_intent",index=self.steps_started,phase=phase,command=command)

    def returned(self):
        if self.steps_started!=self.steps_returned+1:
            raise RuntimeError("unmatched step return")
        self.steps_returned+=1
        self.record("step_return",index=self.steps_returned)

    def sample(self, client, robot, joint_ids, pad_ids, object_ids, phase, measured_faces):
        # Preserve PyBullet's two witness positions, world normal, signed depth,
        # Fn, both tangent directions / forces and exact link/body IDs.
        # Joint tuple: q, dq, reaction wrench, actual last-step motor effort.
        joint_states=client.getJointStates(robot,joint_ids)
        contacts=client.getContactPoints()
        objects=[dict(body=b,pose=client.getBasePositionAndOrientation(b),
                      velocity=client.getBaseVelocity(b)) for b in object_ids]
        self.record('state',index=self.steps_returned,phase=phase,
            joint_ids=joint_ids,joint_states=joint_states,pad_ids=pad_ids,
            contacts=contacts,objects=objects,measured_inner_faces=measured_faces)
        return joint_states,contacts,objects

    def finish(self, **outcome):
        self.stream.close()
        exclusive_json(self.dir/'result.json',dict(**outcome,at=timestamp(),
            step_intents=self.steps_started,step_returns=self.steps_returned,
            unknown_steps=self.steps_started-self.steps_returned,
            telemetry_sha256=digest(self.dir/'telemetry.jsonl.gz')))


def save_rgb(path, width, height, rgba):
    """Lossless RGB camera frame. No dependency on PIL in the frozen environment."""
    data=bytes(rgba)
    if len(data)!=width*height*4:
        raise ValueError("RGBA dimensions disagree")
    rows=bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            start=4*(y*width+x)
            rows.extend(data[start:start+3])
    def chunk(kind,payload):
        return struct.pack('>I',len(payload))+kind+payload+struct.pack('>I',zlib.crc32(kind+payload)&0xffffffff)
    with Path(path).open('xb') as out:
        out.write(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))
            +chunk(b'IDAT',zlib.compress(rows))+chunk(b'IEND',b''))
