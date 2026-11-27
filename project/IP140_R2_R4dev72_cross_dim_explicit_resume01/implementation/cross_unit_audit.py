"""Offline old/new counterfactual audit. The formal selector never imports this module."""
from pathlib import Path
import numpy as np
import json,hashlib,subprocess,sys,os
V=Path(__file__).resolve().parents[1];ROOT=V.parent
OLD=ROOT/'IP140_R2_R4dev72_nonnegative_score_domain_resume01'
def load(p):return json.loads(p.read_bytes())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,obj):
    with p.open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,allow_nan=False,indent=2)
def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def exact(o):return json.dumps(o,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def execution_order(nset):return {s:i for i,s in enumerate(nset)}
def compare_rank(new,ref):
    out={}
    for name in ('source_indices','candidate_camera','seed_pixels','eligibility_mask'):
        with np.load(new/'pool.npz') as a,np.load(ref/'pool.npz') as b:out[name]=bits(a[name],b[name])
    a=load(new/'orders.json');b=load(ref/'orders.json')
    for name in ('scores','orders_by_pool_index','candidate_details','pool_count','G1_volume_m3'):
        out[name]=exact(a[name])==exact(b[name])
    return out
def old_score_audit(directory,network,old):
    # When no archived ranking exists this isolated comparator uses only the old-decoded rows.
    # Persisted files are diagnostic and never referenced as a formal pool by eligibility/intent.
    dest=directory/'forward_audit'/'counterfactual_old_scoring';dest.mkdir()
    meta=load(directory/'network.json')
    with (dest/'network.npz').open('xb') as f:np.savez_compressed(f,**(network|{'candidates':old}))
    meta.update(network_output_sha256=sha(dest/'network.npz'),counterfactual_only=True)
    put(dest/'network.json',meta)
    with (dest/'scoring.log').open('xb') as f:
        ret=subprocess.run([str(ROOT/'runtime/envs/grasp_eval/Scripts/python.exe'),'-B',str(V/'implementation/policy_rank.py'),str(dest)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    if ret.returncode:raise RuntimeError('unexpected shadow ranking failure at '+directory.name)
    return dest
def review(key,ordinal):
    d=V/'units'/key;a=d/'forward_audit';old=OLD/'units'/key
    with np.load(d/'network.npz',allow_pickle=False) as z:new_data={k:z[k] for k in z.files}
    new=new_data['candidates']; prior=np.load(a/'diagnostic_legacy_candidates.npy',allow_pickle=False)
    assert bits(new,np.load(d/'raw_decoder_pool.npy',allow_pickle=False))
    assert new.shape==prior.shape and new.ndim==2 and new.shape[1]==17
    n=len(new)
    indices=np.r_[0:4,13:17]
    assert bits(new[:,indices],prior[:,indices]) and np.isfinite(new).all()
    archival=old/'network.npz'
    historical_check=None
    if ordinal<=8:
        assert archival.is_file()
        expected=[9,5,14,12,8,15,8,3][ordinal-1];assert n==expected
        with np.load(archival,allow_pickle=False) as h:
            historical_check={f:bits(h[f],x) for f,x in new_data.items() if f!='candidates'}
            historical_check['old_17_dim_candidate']=bits(h['candidates'],prior)
            if not all(historical_check.values()):raise ValueError('historical same-seed field drift '+str(historical_check))
    rot=new[:,4:13].astype(float).reshape(-1,3,3);rot0=prior[:,4:13].astype(float).reshape(-1,3,3)
    def residual(r):return float(np.max(abs(r.T@r-np.eye(3)))),float(np.linalg.det(r))
    ledger=load(d/'candidate_domain_ledger.json')['rows'];assert len(ledger)==n
    rankings=load(d/'orders.json');path=None;old_invalid=[]
    for i,r in enumerate(rot0):
        err,det=residual(r)
        if not np.isfinite(r).all() or err>1e-4 or abs(det-1)>1e-4:old_invalid.append(i)
    if n!=3:
        assert bits(new,prior) and not old_invalid
    if (old/'orders.json').exists():
        path=old
    elif not old_invalid:
        path=old_score_audit(d,new_data,prior)
    equality=compare_rank(d,path) if path else None
    if n!=3 and not all(equality.values()):raise ValueError('unexpected source-index/score/order drift '+key+str(equality))
    rows=[]
    with np.load(d/'pool.npz',allow_pickle=False) as pool:
        source=pool['source_indices'];assert bits(new[source],pool['candidate_camera']);slots=execution_order(source.tolist())
    for i,r in enumerate(rot):
        oo,det=residual(r);assert oo<=1e-4 and abs(det-1)<=1e-4
        diff=np.flatnonzero(np.any(new[i].view(np.uint8).reshape(17,-1)!=prior[i].view(np.uint8).reshape(17,-1),axis=1)).tolist()
        assert not set(diff)-set(range(4,13))
        j=slots.get(i)
        rows.append(dict(native_row=i,seed_pixel=int(new_data['pixels'][i]),fields_changed_zero_based=diff,
            old_rotation_residual_and_determinant=residual(rot0[i]),new_rotation_residual_and_determinant=(oo,det),
            old_17_sha256=hashlib.sha256(prior[i].tobytes()).hexdigest(),new_17_sha256=hashlib.sha256(new[i].tobytes()).hexdigest(),
            eligibility_and_exclusions=ledger[i],new_pool_index=j,
            new_ranks_zero_based={m:order.index(j) if j is not None else None for m,order in rankings['orders_by_pool_index'].items()}))
    if ordinal==6:
        assert key=='scene-017-p3-clean' and n==15 and np.flatnonzero(new[:,0]<0).tolist()==[1,10] and len(source)==4
    if ordinal==8:assert key=='scene-021-p1-joint' and old_invalid==[0,1,2]
    return dict(unit_id=key,ordinal=ordinal,N=n,passed=True,historical_archived_raw_available=archival.exists(),
        historical_fields_bit_equal=historical_check,all_declared_nonrotation_fields_bit_equal=True,
        source_identity_audit='forward_audit/row_comparison.json',old_decoder_invalid_rows=old_invalid,
        prior_ranking_reference=path.relative_to(ROOT).as_posix() if path else None,
        prior_ranking_disposition='existing_or_shadow_prior_score' if path else 'not_defined_under_frozen_integrity_gate',
        comparison=equality,native_row_ledger=rows,diagnostic_old_candidate_sha256=sha(a/'diagnostic_legacy_candidates.npy'),
        formal_new_raw_sha256=sha(d/'raw_decoder_pool.npy'),formal_new_pool_sha256=sha(d/'pool.npz'))
