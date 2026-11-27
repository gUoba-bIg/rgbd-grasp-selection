"""Lossless public PyBullet API ledger for one DIRECT session (including camera buffers)."""
import base64
import collections
import gzip
import hashlib
import json
import math
import os
import time
from pathlib import Path

class ApiTrace:
    def __init__(self, module, directory):
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=False)
        self.stream = gzip.open(self.directory/'calls.jsonl.gz','xb',compresslevel=1)
        self.module = module
        self.api = self
        self.counts = collections.Counter()
        self.returns = collections.Counter()
        self.sequence = 0
        self.blobs = {}
        self.exceptions = []
        self.audit_rows = []
        self.audited_sequences = set()
        self.protected_object = None
        self.protected_write_started = False
        self.step_authorized = False
        self.connection_scope = None
        self._authorization = None

    def protect_object(self, body, client):
        if self.protected_object is not None:
            raise RuntimeError("object write guard already armed")
        self.protected_object = (body, client)


    def authorize_physics(self, receipt_path, ledger_path, *, client):
        if self.step_authorized or self._authorization is not None or self.counts['stepSimulation'] or self.connection_scope!='dynamic':
            raise RuntimeError('authorization may occur only once before first physical call')
        receipt_path=Path(receipt_path).resolve();ledger_path=Path(ledger_path).resolve()
        here=Path(__file__).resolve().parent
        if receipt_path!=here/'episodes'/'14'/'precharge.json' or ledger_path!=here/'slot_ledger.json':
            raise RuntimeError('unregistered charge paths')
        self._authorization=dict(receipt_path=receipt_path,ledger_path=ledger_path,client=client,
            receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest())
        self._verify_durable_charge()
        self.step_authorized=True

    def _verify_durable_charge(self):
        a=self._authorization
        if a is None:raise RuntimeError('missing durable authorization')
        receipt_bytes=a['receipt_path'].read_bytes()
        if hashlib.sha256(receipt_bytes).hexdigest()!=a['receipt_sha256']:
            raise RuntimeError('precharge receipt changed')
        receipt=json.loads(receipt_bytes);ledger=json.loads(a['ledger_path'].read_bytes())
        if (receipt.get('slot')!=14 or receipt.get('starts_before')!=8 or receipt.get('starts_after')!=9
            or receipt.get('remaining_after')!=5 or receipt.get('total')!=14
            or receipt.get('client_id')!=a['client'] or receipt.get('independent_confirmation') is not False
            or receipt.get('not_p3') is not True or ledger.get('starts')!=9
            or ledger.get('existing_budget')!=14 or ledger.get('remaining_total_budget')!=5
            or ledger['slots'][13].get('charged') is not True or any(x['charged'] for x in ledger['slots'][7:12])
            or self.protected_object!=(receipt.get('object_body_id'),a['client']) or not self.protected_write_started):
            raise RuntimeError('durable precharge, protected write or allowed slot mismatch')

    def encode(self, value):
        if isinstance(value, bytes):
            return {'bytes_b64':base64.b64encode(value).decode('ascii')}
        if hasattr(value, 'dtype') and hasattr(value,'shape'):
            raw=value.tobytes()
            h=hashlib.sha256(raw).hexdigest()
            if h not in self.blobs:
                with gzip.open(self.directory/(h+'.bin.gz'),'xb',compresslevel=1) as f:
                    f.write(raw)
                self.blobs[h]=len(raw)
            return {'data_sha256':h,'dtype':value.dtype.str,'shape':list(value.shape),'order':'C'}
        raise TypeError(type(value).__name__)

    def write(self, row):
        if row[1] == "call":
            method = row[3]
            if method in {"connect", "loadURDF", "createMultiBody", "changeDynamics", "disconnect", "setCollisionFilterPair"} or (
                    method == "stepSimulation" and self.counts[method] == 1):
                self.audited_sequences.add(row[0])
        if row[0] in self.audited_sequences:
            self.audit_rows.append(row)
        self.stream.write((json.dumps(row,default=self.encode,separators=(',',':'))+'\n').encode('utf-8'))

    def __getattr__(self,name):
        original=getattr(self.module,name)
        if not callable(original):
            return original
        def call(*args,**kwargs):
            if name == 'stepSimulation':
                if not self.step_authorized or self.connection_scope != 'dynamic' or self._authorization is None:
                    raise RuntimeError('physical steps require verified durable slot14 precharge')
                if kwargs.get('physicsClientId') != self._authorization['client'] or self.counts[name]>=4728:
                    raise RuntimeError('wrong client or exhausted single-episode physical budget')
                if not self.counts[name]:
                    self._verify_durable_charge()
            if name in ('createConstraint','resetBasePositionAndOrientation','resetBaseVelocity','resetSimulation','removeBody'):
                raise RuntimeError('body reset, constraint attachment or world replacement forbidden')
            if self.step_authorized and name in ('loadURDF','createMultiBody','changeDynamics','setPhysicsEngineParameter','setGravity','setCollisionFilterPair','setCollisionFilterGroupMask'):
                raise RuntimeError('post-admission scene/physical parameter mutation forbidden')
            if name in ('resetJointState','resetBasePositionAndOrientation','resetBaseVelocity') and (self.step_authorized or self.counts['stepSimulation']):
                raise RuntimeError('dynamic state reset forbidden after first-step authorization')
            if name in ('resetBasePositionAndOrientation','resetBaseVelocity') and self.protected_object is not None:
                obj=args[0] if args else kwargs.get('bodyUniqueId')
                if obj==self.protected_object[0]:raise RuntimeError('object state reset forbidden')
            if name == 'changeDynamics' and self.protected_object is not None:
                body = args[0] if args else kwargs.get('bodyUniqueId')
                if body == self.protected_object[0]:
                    if self.protected_write_started:
                        raise RuntimeError("a second dynamics write to object 021 is forbidden")
                    self.protected_write_started = True
            self.sequence+=1
            seq=self.sequence
            self.counts[name]+=1
            self.write([seq,'call',time.time_ns(),name,args,kwargs])
            if name == 'stepSimulation' and self.counts[name] == 1:
                # Charge and this call intent survive before entering the native engine.
                self.stream.flush()
                os.fsync(self.stream.fileobj.fileno())
            try:
                result=original(*args,**kwargs)
            except BaseException as ex:
                self.exceptions.append([seq,name,repr(ex)])
                self.write([seq,'error',time.time_ns(),repr(ex)])
                self.stream.flush()
                raise
            self.write([seq,'return',time.time_ns(),result])
            self.returns[name]+=1
            if name.startswith('setCollisionFilter') or name in ('connect','loadURDF','disconnect'):
                self.stream.flush()
            return result
        call.__name__=name
        setattr(self,name,call)
        return call

    def close(self):
        self.stream.close()
        with (self.directory/'summary.json').open('x',encoding='utf-8') as f:
            json.dump(dict(calls=self.counts,returns=self.returns,exceptions=self.exceptions,
                total_calls=self.sequence,lossless_raw_blobs_bytes=self.blobs,
                timestamps='Unix epoch ns; call and return share same sequence ID'),f,indent=2)
            f.write('\n')
