"""Lossless public PyBullet API ledger for one DIRECT session (including camera buffers)."""
import base64
import collections
import gzip
import hashlib
import json
import math
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

    def protect_object(self, body, client):
        if self.protected_object is not None:
            raise RuntimeError("object write guard already armed")
        self.protected_object = (body, client)


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
                raise RuntimeError("smoke14 prohibits every physical step; no dynamic authorization path")
            if name in ('createConstraint','resetBasePositionAndOrientation','resetBaseVelocity'): raise RuntimeError('smoke14 forbids body teleport, twist reset and new constraints')
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
