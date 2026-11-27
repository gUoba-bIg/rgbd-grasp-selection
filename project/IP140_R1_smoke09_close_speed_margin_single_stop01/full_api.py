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
        self.stream.write((json.dumps(row,default=self.encode,separators=(',',':'))+'\n').encode('utf-8'))

    def __getattr__(self,name):
        original=getattr(self.module,name)
        if not callable(original):
            return original
        def call(*args,**kwargs):
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
