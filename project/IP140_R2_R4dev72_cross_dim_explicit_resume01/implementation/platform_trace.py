"""Lossless public PyBullet API ledger for one DIRECT session (including camera buffers)."""
import base64
import collections
import gzip
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from platform_imports import VERSION

class R4Trace:
    def __init__(self, module, directory, scope):
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
        if scope not in ('observation','dynamic','contact_aware_static','eligibility'):
            raise ValueError('unknown native connection scope')
        self.connection_scope=scope
        self.limit=240 if scope=='observation' else 4728 if scope=='dynamic' else 0
        self.observation_steps=0
        self._authorization = None
        self.actual_step_calls = 0

    def protect_object(self, body, client):
        if self.protected_object is not None:
            raise RuntimeError("object write guard already armed")
        self.protected_object = (body, client)


    def authorize_observation(self, client=None):
        if self.connection_scope!='observation' or not self.protected_object or self.counts['stepSimulation']:
            raise RuntimeError('unregistered observation')
        if client is None: client=self.protected_object[1]
        if client!=self.protected_object[1] or self.step_authorized or not self.protected_write_started:
            raise RuntimeError('observation setup incomplete')
        self._authorization=dict(client=client,receipt_path=self.directory/'observation.json',on_first_observed=lambda *args:None)
        self.step_authorized=True

    def authorize_dynamic(self,client,*,intent_path,intent_sha256,static_receipt):
        if self.connection_scope!='dynamic' or self.step_authorized or self.counts['stepSimulation']:
            raise RuntimeError('one sealed method only')
        if not self.protected_object or self.protected_object[1]!=client or not self.protected_write_started:
            raise RuntimeError('method physics object not bound')
        self._authorization=dict(client=client,receipt_path=Path(intent_path),intent_sha256=intent_sha256,
               static_receipt=Path(static_receipt),on_first_observed=lambda *args:None)
        self._verify_reservation()
        self.step_authorized=True

    def _verify_reservation(self):
        if self.counts['setCollisionFilterPair']!=14 or self.counts['connect']!=1 or self.exceptions:
            raise RuntimeError('required audited filters or native session not verified')
        if self.connection_scope=='dynamic':
            a=self._authorization
            if hashlib.sha256(a['receipt_path'].read_bytes()).hexdigest()!=a['intent_sha256']:
                raise RuntimeError('frozen intent changed')
            r=json.loads(a['static_receipt'].read_bytes())
            if r.get('passed') is not True or r.get('intent_sha256')!=a['intent_sha256'] or r.get('static_steps')!=0:
                raise RuntimeError('post-lock zero-step safety certificate not bound')

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
                if not self.step_authorized or self.connection_scope not in ('observation','dynamic') or self._authorization is None:
                    raise RuntimeError('physical steps require verified slot reservation')
                if kwargs.get('physicsClientId') != self._authorization['client'] or self.counts[name]>=self.limit:
                    raise RuntimeError('wrong client or exhausted single-episode physical budget')
                if not self.counts[name]:
                    self._verify_reservation()
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
                        raise RuntimeError("a second dynamics write to the registered object is forbidden")
                    self.protected_write_started = True
            self.sequence+=1
            seq=self.sequence
            self.counts[name]+=1
            self.write([seq,'call',time.time_ns(),name,args,kwargs])
            if name == 'stepSimulation' and self.counts[name] == 1:
                # This pre-call record is intent only; the ledger retains the previously observed start count.
                self.stream.flush()
                os.fsync(self.stream.fileobj.fileno())
            first_physical = name == "stepSimulation" and self.actual_step_calls == 0
            # Capture the CPython C-call dispatch separately from the wrapper's
            # pre-call intent, including a hard process loss inside native code.
            if first_physical:
                if sys.getprofile() is not None:
                    raise RuntimeError("unavailable first-call dispatch observer")
                def dispatch(frame,event,fn):
                    if event == "c_call" and fn is original:
                        path = self.directory/"native_call_dispatch.json"
                        data = dict(scope=self.connection_scope,api_sequence=seq,method=name,physicsClientId=kwargs["physicsClientId"],
                                    event="CPython c_call",dispatch_time_ns=time.time_ns(),native_return_observed=False)
                        with path.open("x",encoding="utf-8") as out:
                            json.dump(data,out);out.flush();os.fsync(out.fileno())
                sys.setprofile(dispatch)
            try:
                result=original(*args,**kwargs)
            except BaseException as ex:
                if first_physical:
                    sys.setprofile(None)
                self.exceptions.append([seq,name,repr(ex)])
                if name == "stepSimulation":
                    self.actual_step_calls += 1
                    if first_physical:
                        self._authorization["on_first_observed"](seq,False,repr(ex))
                self.write([seq,'error',time.time_ns(),repr(ex)])
                self.stream.flush()
                raise
            if first_physical:
                sys.setprofile(None)
            if name == "stepSimulation":
                self.actual_step_calls += 1
                self.returns[name]+=1
                if first_physical:
                    self._authorization["on_first_observed"](seq,True,None)
            self.write([seq,'return',time.time_ns(),result])
            if name != "stepSimulation":
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
                total_calls=self.sequence,actual_native_step_calls=self.actual_step_calls,lossless_raw_blobs_bytes=self.blobs,
                timestamps='Unix epoch ns; call and return share same sequence ID'),f,indent=2)
            f.write('\n')
