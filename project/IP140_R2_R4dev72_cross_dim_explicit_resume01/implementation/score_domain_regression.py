"""Independent pre-seal comparisons on raw, rebuilt and historical outputs."""
from pathlib import Path
import hashlib,json
import numpy as np

V=Path(__file__).resolve().parents[1]
OLD=V.parent/'IP140_R2_R4dev72_raw_a_c0_paired01'
def read(p):return json.loads(p.read_bytes())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bytes_equal(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def exact(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def review(unit,ordinal):
    p=V/'units'/unit
    old=OLD/'units'/unit
    new=read(p/'orders.json')
    with np.load(p/'network.npz',allow_pickle=False) as n,np.load(p/'pool.npz',allow_pickle=False) as c:
        raw=n['candidates'];idx=c['source_indices'];pool=c['candidate_camera']
        assert bytes_equal(raw[idx],pool)
        bounds=(raw[:,0]>=0)&(raw[:,1:4]>0).all(axis=1)&(raw[:,1]<=np.asarray(.1,dtype=raw.dtype))
        assert np.isfinite(raw).all() and bounds[idx].all()
        assert len(set(idx.tolist()))==len(idx)==new['pool_count']
        assert new['scores']['Raw']==pool[:,0].astype(float).tolist()
        assert all(sorted(o)==list(range(len(idx))) for o in new['orders_by_pool_index'].values())
        result=dict(unit_id=unit,ordinal=ordinal,raw_rows=len(raw),source_network_sha256=sha(p/'network.npz'),
                    pool_count=len(idx),eligible_source_indices=idx.tolist(),all_eligible_rows_native_bytes_identical=True,
                    raw_negative_source_indices=np.nonzero(raw[:,0]<0)[0].tolist(),immutable_zero_score_sign_bits=np.signbit(raw[raw[:,0]==0,0]).tolist())
        if ordinal<=6:
            assert sha(old/'network.npz')==sha(p/'network.npz') and sha(old/'network.json')==sha(p/'network.json')
            with np.load(old/'pool.npz',allow_pickle=False) as prev:
                oldidx=prev['source_indices']
                if ordinal<=5:
                    assert bytes_equal(prev['candidate_camera'],pool) and bytes_equal(oldidx,idx)
                else:
                    assert unit=='scene-017-p3-clean' and len(raw)==15 and len(oldidx)==5 and len(idx)==4
                    assert list(oldidx[bounds[oldidx]])==list(idx) and (raw[:,0]<0).sum()==2
                    result['old_pool_source_indices']=oldidx.tolist()
                    result['removed_original_native_scores']=[float(raw[i,0]) for i in result['raw_negative_source_indices']]
            if ordinal<=5:
                previous=read(old/'orders.json')
                result['bit_exact_fields']=[]
                for field in ('candidate_details','scores','orders_by_pool_index'):
                    assert exact(previous[field])==exact(new[field]),(unit,field)
                    result['bit_exact_fields'].append(field)
        result['passed']=True
        return result
